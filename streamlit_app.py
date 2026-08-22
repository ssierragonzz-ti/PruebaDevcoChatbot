"""
MacBot - Asistente virtual para una tienda de productos Apple.

Arquitectura:
- El cliente HTTP hacia la API de Gemini es un recurso compartido (sin
  estado propio de conversación) cacheado a nivel de proceso.
- Cada visitante tiene su propia sesión de chat, guardada en
  st.session_state, para no mezclar conversaciones entre usuarios.
- El modelo activo se resuelve una sola vez contra una lista de
  candidatos, para tolerar que Google retire o renombre modelos.
- El catálogo/precios puede vivir en una hoja de Google Sheets publicada
  como CSV (para editarlo sin redeployar); si no está configurada, se usa
  un catálogo embebido en el código como respaldo. El bot nunca se queda
  sin catálogo.
- El log de conversaciones puede persistir en una hoja de Google Sheets
  (si hay credenciales de service account en Secrets); si no, cae de
  vuelta al CSV local no persistente (comportamiento original).
- Los "botones de interés" del sidebar responden de dos formas distintas
  a propósito:
    * instantáneas -> se contestan con texto fijo, SIN llamar a Gemini
      (costo cero, cero riesgo de alucinación, no cuentan para el límite
      de mensajes por sesión).
    * llm -> se mandan como si el usuario las hubiera escrito, para que
      el modelo razone (recomendaciones, comparaciones).
  Este patrón híbrido (menú determinístico + LLM solo donde aporta) es
  el mismo que usan la mayoría de bots de atención al cliente de retail
  (catálogos de WhatsApp Business, Intercom Fin, etc.) y es lo que hace
  viable operar esto a bajo costo.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import streamlit as st
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

# =============================================================================
# CONFIGURACIÓN Y CONSTANTES
# =============================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("macbot")

# Modelos candidatos, del más nuevo al más antiguo. Google retira modelos de
# Gemini con frecuencia; en vez de fijar un solo nombre, probamos esta lista
# en orden y usamos el primero que responda.
#
# ⚠️ REVISAR ANTES DEL 16 DE OCTUBRE DE 2026: Google anunció el retiro de
# toda la serie gemini-2.5-* (Gemini Developer API) para esa fecha. Si en
# ese momento esta lista todavía termina en "gemini-2.5-flash-lite" como
# único respaldo, hay que reemplazarlo por un modelo 3.x vigente.
CANDIDATE_MODELS: tuple[str, ...] = (
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.6-flash",
    "gemini-2.5-flash-lite",
)

MAX_RETRIES = 3
RETRY_BASE_DELAY_SECONDS = 1.0
RETRYABLE_STATUS_CODES = {429, 500, 502, 503}

# Límite blando de mensajes por sesión: protege la cuenta de Gemini de un
# uso ilimitado por visitante. No es un límite de seguridad, es de costo.
# Las respuestas instantáneas (botones de info fija) NO cuentan para este
# límite porque no llaman a la API.
MAX_MENSAJES_POR_SESION = int(os.getenv("MACBOT_MAX_MENSAJES_SESION", "40"))

LOG_FILE = os.getenv("MACBOT_LOG_FILE", "logs.csv")
LOG_CSV_COLUMNAS = ["fecha_utc", "modelo", "mensaje_usuario", "respuesta_bot"]
_log_lock = threading.Lock()  # protege escrituras concurrentes de varios usuarios

CATALOGO_COLUMNAS_REQUERIDAS = ["categoria", "producto", "detalle", "precio_texto"]

# Catálogo de respaldo: se usa si CATALOG_SHEET_CSV_URL no está configurada,
# o si la hoja no se puede leer por cualquier motivo (URL caída, columnas
# incorrectas, hoja vacía). El bot nunca debe quedarse sin catálogo.
CATALOGO_EMBEBIDO: list[dict[str, str]] = [
    {"categoria": "iPhone", "producto": "iPhone 15", "detalle": "chip A16, cámara 48 MP, USB-C", "precio_texto": "Desde $4,299,000"},
    {"categoria": "iPhone", "producto": "iPhone 15 Pro", "detalle": "chip A17 Pro, titanio, teleobjetivo 48 MP", "precio_texto": "Desde $5,499,000"},
    {"categoria": "iPhone", "producto": "iPhone 16", "detalle": "chip A18, cámara Fusion, botón Acción y Control de Cámara", "precio_texto": "Desde $4,799,000"},
    {"categoria": "iPhone", "producto": "iPhone 16 Pro", "detalle": "chip A18 Pro, pantalla más grande, grabación espacial", "precio_texto": "Desde $5,999,000"},

    {"categoria": "Mac", "producto": "MacBook Air M3", "detalle": '13.6", 8 GB RAM, 256 GB SSD', "precio_texto": "Desde $5,999,000"},
    {"categoria": "Mac", "producto": "MacBook Pro M3 Pro", "detalle": '14", 18 GB RAM, 512 GB SSD', "precio_texto": "Desde $11,999,000"},
    {"categoria": "Mac", "producto": "Mac mini M4", "detalle": "compacto, ideal para escritorio con monitor propio", "precio_texto": "Desde $3,999,000"},
    {"categoria": "Mac", "producto": "iMac M4", "detalle": '24", todo en uno, colores vivos', "precio_texto": "Desde $6,999,000"},
    {"categoria": "Mac", "producto": "Mac Studio", "detalle": "estaciones de trabajo de alto rendimiento (edición de video, 3D)", "precio_texto": "Desde $18,999,000"},

    {"categoria": "iPad", "producto": "iPad (básico)", "detalle": 'pantalla 10.9", ideal para uso diario y estudio', "precio_texto": "Desde $2,199,000"},
    {"categoria": "iPad", "producto": "iPad Air", "detalle": "chip M2, más potencia para diseño e ilustración", "precio_texto": "Desde $3,399,000"},
    {"categoria": "iPad", "producto": "iPad Pro", "detalle": "pantalla OLED, con Apple Pencil Pro", "precio_texto": "Desde $6,299,000"},
    {"categoria": "iPad", "producto": "iPad mini", "detalle": "portátil, buena para lectura y notas rápidas", "precio_texto": "Desde $2,899,000"},

    {"categoria": "Accesorios y otros", "producto": "AirPods Pro 2", "detalle": "cancelación activa de ruido, chip H2", "precio_texto": "$1,299,000"},
    {"categoria": "Accesorios y otros", "producto": "AirPods 4", "detalle": "nuevos, con o sin cancelación de ruido", "precio_texto": "Desde $799,000"},
    {"categoria": "Accesorios y otros", "producto": "Apple Watch Series 9", "detalle": "41 mm, GPS", "precio_texto": "$2,399,000"},
    {"categoria": "Accesorios y otros", "producto": "Apple Watch Ultra 2", "detalle": "resistente, GPS + Cellular, para deporte extremo", "precio_texto": "$6,999,000"},
    {"categoria": "Accesorios y otros", "producto": "Apple TV 4K", "detalle": "streaming, ideal para complementar un Mac o iPhone", "precio_texto": "$1,499,000"},
    {"categoria": "Accesorios y otros", "producto": "HomePod mini", "detalle": "altavoz inteligente, se integra con el resto del ecosistema", "precio_texto": "$599,000"},
    {"categoria": "Accesorios y otros", "producto": "AirTag", "detalle": "localizador de objetos", "precio_texto": "$149,000 (pack x4: $499,000)"},
    {"categoria": "Accesorios y otros", "producto": "Fundas, cargadores y cables originales", "detalle": "", "precio_texto": "precios desde $69,000"},
]


# =============================================================================
# DATOS DE LA TIENDA (fuente única: se usan tanto en el prompt del LLM como
# en las respuestas instantáneas de los botones, para que nunca queden
# desincronizados entre sí).
# =============================================================================

@dataclass(frozen=True)
class InfoTienda:
    nombre: str
    horario_texto: str
    direccion: str
    whatsapp_numero: str  # solo dígitos, con código de país, sin '+' ni espacios
    web: str
    instagram: str

    @property
    def whatsapp_link(self) -> str:
        return f"https://wa.me/{self.whatsapp_numero}"


# ⚠️ Editar estos valores con los datos reales de la tienda.
INFO_TIENDA = InfoTienda(
    nombre="MacStore Ejemplo",
    horario_texto="Lunes a sábado 9:00-20:00, domingo 10:00-18:00 (festivos: 10:00-17:00).",
    direccion="Centro Comercial X, local 123, Bogotá.",
    whatsapp_numero="573000000000",
    web="www.macstore-ejemplo.com",
    instagram="@macstoreejemplo",
)

SYSTEM_PROMPT_TEMPLATE = """
Eres "MacBot", el asistente virtual de {nombre_tienda}, una tienda
especializada en productos Apple. Respondes de forma amable, clara y
breve (máximo 3 párrafos), con emojis moderados.

## Datos de la tienda
- Horario: {horario}
- Dirección: {direccion}
- WhatsApp: {whatsapp_link}
- Web: {web}
- Redes: {instagram} en Instagram

## Catálogo destacado
{catalogo}

## Recomendaciones según necesidad
- Fotografía / contenido: iPhone 16 Pro o iPhone 15 Pro.
- Diseño gráfico / edición de video: MacBook Pro o Mac Studio.
- Estudio, oficina, uso general: MacBook Air o iPad Air.
- Toma de notas y lectura: iPad o iPad mini con Apple Pencil.
- Deporte y salud: Apple Watch (Ultra 2 si es deporte de alto rendimiento).
- Hogar inteligente: HomePod mini + Apple TV 4K.
- Evitar perder cosas: AirTag.
- Si el catálogo fue editado y alguno de estos modelos ya no aparece,
  prioriza siempre lo que sí esté en el catálogo vigente arriba.
- Si piden comparar dos o más productos del catálogo, arma una
  comparación breve en viñetas (precio, para quién es mejor cada uno).
- Si preguntan por un producto Apple que no aparece en el catálogo
  vigente (por ejemplo, un modelo descontinuado o uno que la tienda no
  maneja), acláralo con honestidad y ofrece la alternativa más cercana
  del catálogo actual, o deriva a WhatsApp para confirmar disponibilidad.

## Políticas
- Garantía: 1 año con Apple en todos los productos; AppleCare+ opcional (cobertura extendida y daños accidentales).
- Devoluciones: 30 días en empaque original con factura.
- Reparaciones: diagnóstico gratuito, solo repuestos originales, en centro de servicio autorizado. Fuera de garantía, el costo de repuestos y mano de obra se cotiza después del diagnóstico, no antes.
- Envíos: Bogotá mismo día (pedidos antes de las 3pm), otras ciudades 1-2 días hábiles.
- Financiación: hasta 24 cuotas con tarjetas de crédito aliadas; consulta bancos participantes por WhatsApp.
- Formas de pago en tienda: efectivo, tarjeta débito/crédito y medios digitales habituales (confirmar el detalle exacto por WhatsApp, puede variar).
- Apartado de producto: se puede reservar con un abono; condiciones exactas por WhatsApp.
- Plan de renovación (trade-in): recibimos tu equipo Apple usado como parte de pago; el valor depende del modelo y estado, se evalúa en tienda o por fotos vía WhatsApp.
- Descuento estudiantes: aplica con carné vigente, consultar vigencia y porcentaje actual por WhatsApp.

## Reglas de conversación
- Si preguntan por disponibilidad o stock exacto: no tienes acceso al inventario en tiempo real, sugiere WhatsApp para confirmar.
- Si el cliente quiere comprar o cerrar un pedido: ofrece el enlace de WhatsApp.
- Si es fuera del horario de atención: pide nombre y teléfono para que un asesor lo contacte al día siguiente.
- Si preguntan algo técnico que no es de la tienda (ej. cómo resetear un iPhone, cómo actualizar iOS): puedes dar una respuesta general y breve, pero aclara que para casos específicos lo mejor es pasar por el centro de servicio o el soporte oficial de Apple.
- No inventes precios, stock ni características que no estén aquí. Si no sabes algo, dilo y deriva a un humano.
- Nunca pidas datos sensibles como números de tarjeta, claves o contraseñas.
- Si el mensaje no tiene relación con la tienda ni con productos Apple, redirige amablemente la conversación.
- Responde siempre en el mismo idioma en el que te escribe el visitante (si escribe en inglés, respondes en inglés, manteniendo el mismo tono).
"""

# Respuestas fijas para los botones "instantáneos": no llaman a Gemini, así
# que tienen costo cero y no pueden alucinar datos de la tienda. Se
# construyen a partir de INFO_TIENDA para no repetir los datos a mano.
CANNED_ANSWERS: dict[str, str] = {
    "horario": (
        f"🕒 Nuestro horario es: {INFO_TIENDA.horario_texto}\n\n"
        f"📍 Nos encuentras en: {INFO_TIENDA.direccion}"
    ),
    "garantia": (
        "🛡️ Todos los productos tienen 1 año de garantía con Apple. "
        "Puedes agregar AppleCare+ para cobertura extendida y daños accidentales.\n\n"
        "**Devoluciones:** 30 días, en empaque original y con factura.\n\n"
        "**Reparaciones fuera de garantía:** el diagnóstico es gratuito; el "
        "costo de repuestos y mano de obra se cotiza después de revisar el equipo."
    ),
    "envios": (
        "🚚 **Bogotá:** mismo día si el pedido es antes de las 3:00pm.\n"
        "🚚 **Otras ciudades:** 1-2 días hábiles.\n\n"
        "También puedes recoger en tienda si prefieres apartar el producto primero."
    ),
    "financiacion": (
        "💳 Financiamos hasta 24 cuotas con tarjetas de crédito aliadas, además de "
        "pago en efectivo, tarjeta débito/crédito y medios digitales habituales.\n\n"
        f"Para confirmar bancos y condiciones exactas, escríbenos por WhatsApp: {INFO_TIENDA.whatsapp_link}"
    ),
    "trade_in": (
        "🔄 Recibimos tu equipo Apple usado como parte de pago (plan renove). "
        "El valor depende del modelo y el estado; lo evaluamos en tienda o con "
        "fotos por WhatsApp.\n\n"
        f"Cotízalo aquí: {INFO_TIENDA.whatsapp_link}"
    ),
}


@dataclass(frozen=True)
class AccionRapida:
    """Un botón del sidebar.

    tipo="instantanea": responde con CANNED_ANSWERS[valor], sin llamar a
    Gemini (gratis, no cuenta para el límite de mensajes por sesión).
    tipo="llm": envía `valor` como si el usuario lo hubiera escrito, para
    que el modelo razone (recomendaciones, comparaciones, etc.).
    """
    etiqueta: str
    tipo: Literal["instantanea", "llm"]
    valor: str


QUICK_ACTIONS: tuple[AccionRapida, ...] = (
    AccionRapida("🕒 Horario y ubicación", "instantanea", "horario"),
    AccionRapida("🛡️ Garantía y devoluciones", "instantanea", "garantia"),
    AccionRapida("🚚 Envíos", "instantanea", "envios"),
    AccionRapida("💳 Financiación y pagos", "instantanea", "financiacion"),
    AccionRapida("🔄 Plan renove (trade-in)", "instantanea", "trade_in"),
    AccionRapida("📱 ¿Qué iPhone me recomiendas?", "llm", "¿Qué iPhone me recomiendas según mi uso? Pregúntame para qué lo necesito."),
    AccionRapida("⚖️ Comparar dos productos", "llm", "Quiero comparar dos productos del catálogo, ayúdame a elegir cuáles comparar según mi uso."),
    AccionRapida("🎓 Descuento de estudiante", "llm", "¿Cómo funciona el descuento de estudiante?"),
)


# =============================================================================
# TIPOS
# =============================================================================

@dataclass(frozen=True)
class ModeloResuelto:
    nombre: str


# =============================================================================
# SECRETS (helper compartido: Streamlit Secrets primero, env var de respaldo)
# =============================================================================

def _leer_secreto(nombre: str) -> str | None:
    """Lee un secreto de Streamlit o, si no existe, de una variable de entorno.

    Envuelto en try/except porque acceder a st.secrets puede lanzar una
    excepción (no solo devolver None) si no existe ningún secrets.toml en
    el entorno, por ejemplo al correr localmente sin haberlo configurado.
    """
    try:
        valor = st.secrets.get(nombre)
    except Exception:
        valor = None
    return valor or os.getenv(nombre)


# =============================================================================
# CATÁLOGO (editable desde Google Sheets, sin necesidad de redeploy)
# =============================================================================

@st.cache_data(ttl=600, show_spinner=False)
def cargar_catalogo() -> tuple[list[dict[str, str]], str, str | None]:
    """
    Carga el catálogo desde una hoja de Google Sheets publicada como CSV
    (Archivo > Compartir > Publicar en la Web > CSV), cuya URL se configura
    en el secreto/variable CATALOG_SHEET_CSV_URL. Si no está configurada, o
    la carga falla por cualquier motivo (URL caída, columnas incorrectas,
    hoja vacía), se usa el catálogo embebido en el código como respaldo.

    Devuelve (filas, origen, motivo_fallback):
    - origen en {"sheet", "embebido"}, para mostrarlo en la interfaz.
    - motivo_fallback: None si todo salió bien (o si simplemente no hay URL
      configurada); si no, un texto corto explicando por qué se usó el
      catálogo embebido en vez de la hoja (útil para el panel de admin,
      p. ej. cuando la hoja está publicada pero todavía no tiene filas).
    """
    url = _leer_secreto("CATALOG_SHEET_CSV_URL")
    if not url:
        return CATALOGO_EMBEBIDO, "embebido", None
    try:
        with urllib.request.urlopen(url, timeout=10) as respuesta:
            contenido = respuesta.read().decode("utf-8-sig")
        lector = csv.DictReader(io.StringIO(contenido))
        columnas = set(lector.fieldnames or [])
        faltantes = set(CATALOGO_COLUMNAS_REQUERIDAS) - columnas
        if faltantes:
            raise ValueError(f"faltan columnas en la hoja: {sorted(faltantes)}")
        filas = [
            {c: (fila.get(c) or "").strip() for c in CATALOGO_COLUMNAS_REQUERIDAS}
            for fila in lector
        ]
        filas = [f for f in filas if f["producto"]]
        if not filas:
            raise ValueError("la hoja está publicada pero todavía no tiene filas de productos")
        return filas, "sheet", None
    except (urllib.error.URLError, ValueError, TimeoutError, OSError) as e:
        motivo = str(e)
        logger.warning(
            "No se pudo cargar el catálogo desde la hoja de cálculo (%s). "
            "Se usa el catálogo embebido como respaldo.", motivo,
        )
        return CATALOGO_EMBEBIDO, "embebido", motivo


def formatear_catalogo(filas: list[dict[str, str]]) -> str:
    """Agrupa las filas del catálogo por categoría y arma el bloque de texto
    que se inserta en el prompt de sistema."""
    categorias: dict[str, list[dict[str, str]]] = {}
    for fila in filas:
        categorias.setdefault(fila["categoria"], []).append(fila)

    bloques = []
    for categoria, productos in categorias.items():
        lineas = []
        for p in productos:
            detalle = p["detalle"].strip()
            precio = p["precio_texto"].strip()
            if detalle:
                lineas.append(f"- {p['producto']}: {detalle}. {precio}.")
            else:
                lineas.append(f"- {p['producto']}: {precio}.")
        bloques.append(f"{categoria}:\n" + "\n".join(lineas))
    return "\n\n".join(bloques)


# =============================================================================
# CAPA DE INFRAESTRUCTURA (cacheada a nivel de proceso, sin estado de sesión)
# =============================================================================

def _leer_api_key() -> str:
    api_key = _leer_secreto("GEMINI_API_KEY")
    if not api_key:
        st.error(
            "Falta la API key de Gemini. Agrégala en Settings > Secrets "
            "como GEMINI_API_KEY = \"tu_clave\"."
        )
        st.stop()
    return api_key


@st.cache_resource(show_spinner=False)
def obtener_cliente() -> genai.Client:
    """
    Crea el cliente de la API una única vez por proceso y lo reutiliza en
    todas las sesiones. Recrearlo en cada rerun de Streamlit deja conexiones
    HTTP a medio cerrar y provoca errores tipo "client has been closed".
    """
    return genai.Client(api_key=_leer_api_key())


@st.cache_resource(show_spinner=False)
def resolver_modelo(_cliente: genai.Client) -> ModeloResuelto:
    """
    Prueba los modelos candidatos, en orden, con una llamada mínima y
    sin estado. Se cachea el nombre del primero que responda.

    Se capturan excepciones genéricas (no solo APIError): un fallo de red
    o de transporte al probar un modelo no debería tumbar toda la
    resolución, sino simplemente hacer que se pruebe el siguiente
    candidato de la lista.
    """
    ultimo_error: Exception | None = None
    for nombre in CANDIDATE_MODELS:
        try:
            _cliente.models.generate_content(
                model=nombre,
                contents="ping",
                config=types.GenerateContentConfig(max_output_tokens=5),
            )
            logger.info("Modelo activo resuelto: %s", nombre)
            return ModeloResuelto(nombre=nombre)
        except Exception as e:
            logger.warning("Modelo %s no disponible (%s), probando el siguiente.", nombre, e)
            ultimo_error = e
            continue

    raise RuntimeError(
        f"Ninguno de los modelos candidatos respondió. Último error: {ultimo_error}"
    )


# =============================================================================
# CAPA DE SESIÓN (una conversación por visitante)
# =============================================================================

def construir_generation_config() -> types.GenerateContentConfig:
    """
    Arma la configuración de generación con el prompt de sistema vigente,
    incluyendo el catálogo (de la hoja de cálculo o el embebido).

    Nota de auditoría: temperature/top_p/top_k se dejan fuera a propósito.
    Google los marcó como deprecados y sin efecto en toda la línea 3.x de
    Gemini Flash (incluida gemini-3.5-flash-lite), y advierte que
    generaciones futuras de modelos los rechazarán con error. El control
    de tono/consistencia se hace vía system_instruction, que es lo que
    ahora recomienda Google como reemplazo.
    """
    filas, _origen, _motivo = cargar_catalogo()
    catalogo_texto = formatear_catalogo(filas)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        catalogo=catalogo_texto,
        nombre_tienda=INFO_TIENDA.nombre,
        horario=INFO_TIENDA.horario_texto,
        direccion=INFO_TIENDA.direccion,
        whatsapp_link=INFO_TIENDA.whatsapp_link,
        web=INFO_TIENDA.web,
        instagram=INFO_TIENDA.instagram,
    )
    return types.GenerateContentConfig(
        system_instruction=system_prompt,
        max_output_tokens=500,
    )


def obtener_chat_de_sesion(cliente: genai.Client, modelo: ModeloResuelto):
    """Crea (o reutiliza) la sesión de chat propia de este usuario.

    El catálogo queda congelado en el prompt de sistema en el momento en
    que se crea la sesión: si alguien edita la hoja de cálculo a mitad de
    una conversación ya abierta, esa conversación sigue viendo el catálogo
    que tenía al iniciar. Las conversaciones nuevas sí ven el cambio,
    sujeto al caché de 10 minutos de cargar_catalogo().

    Si crear la sesión falla (red, cuota, etc.) se muestra un error
    amigable en vez de dejar que Streamlit rompa con una traza cruda.
    """
    if "chat" not in st.session_state:
        try:
            config = construir_generation_config()
            st.session_state.chat = cliente.chats.create(model=modelo.nombre, config=config)
        except Exception as e:
            logger.error("No se pudo crear la sesión de chat (%s).", e)
            st.error(
                "No se pudo iniciar el asistente en este momento. "
                "Recarga la página o intenta de nuevo en unos minutos."
            )
            st.stop()
    return st.session_state.chat


def extraer_texto(respuesta) -> str:
    """
    Extrae el texto de una respuesta de Gemini de forma defensiva.
    Una respuesta puede venir sin candidatos si el filtro de seguridad
    bloqueó el contenido; en ese caso no se debe intentar leer .text
    directamente porque puede lanzar una excepción.
    """
    if not getattr(respuesta, "candidates", None):
        return (
            "No puedo responder eso. ¿Quieres preguntarme sobre productos, "
            "garantías, envíos o precios?"
        )
    return respuesta.text


def enviar_con_reintentos(chat, mensaje: str) -> str:
    """Envía un mensaje con reintento y backoff exponencial ante errores
    429/5xx, y también ante errores inesperados de red/transporte (no solo
    APIError), para no dejar sin respuesta al usuario por un fallo puntual
    de conexión."""
    espera = RETRY_BASE_DELAY_SECONDS
    for intento in range(1, MAX_RETRIES + 1):
        try:
            respuesta = chat.send_message(mensaje)
            return extraer_texto(respuesta)
        except genai_errors.APIError as e:
            codigo = getattr(e, "code", None)
            es_reintentable = codigo in RETRYABLE_STATUS_CODES
            logger.warning("Intento %s/%s falló (código=%s): %s", intento, MAX_RETRIES, codigo, e)
            if es_reintentable and intento < MAX_RETRIES:
                time.sleep(espera)
                espera *= 2
                continue
            return f"Ocurrió un error al contactar al asistente ({codigo or 'desconocido'}). Intenta de nuevo en unos segundos."
        except Exception as e:
            logger.warning("Intento %s/%s falló con un error inesperado: %s", intento, MAX_RETRIES, e)
            if intento < MAX_RETRIES:
                time.sleep(espera)
                espera *= 2
                continue
            return "Ocurrió un error inesperado al contactar al asistente. Intenta de nuevo en unos segundos."
    return "No pude obtener respuesta en este momento. Intenta de nuevo en unos segundos."


# =============================================================================
# LOG DE CONVERSACIONES (Google Sheets si está configurado, si no CSV local)
# =============================================================================

@st.cache_resource(show_spinner=False)
def _obtener_hoja_log():
    """
    Si hay credenciales de service account y un ID de hoja en Secrets
    (GOOGLE_SERVICE_ACCOUNT_JSON, LOG_SHEET_ID), devuelve la referencia a
    la hoja para loguear ahí. Si falta cualquiera de los dos, o falla la
    conexión (paquetes no instalados, credenciales inválidas, hoja no
    compartida con el service account, etc.), devuelve None y el llamador
    cae de vuelta al CSV local. Nunca lanza una excepción hacia afuera.
    """
    creds_raw = _leer_secreto("GOOGLE_SERVICE_ACCOUNT_JSON")
    sheet_id = _leer_secreto("LOG_SHEET_ID")
    if not creds_raw or not sheet_id:
        return None
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        info = json.loads(creds_raw) if isinstance(creds_raw, str) else dict(creds_raw)
        alcance = ["https://www.googleapis.com/auth/spreadsheets"]
        credenciales = Credentials.from_service_account_info(info, scopes=alcance)
        cliente_sheets = gspread.authorize(credenciales)
        hoja = cliente_sheets.open_by_key(sheet_id).sheet1
        if not hoja.row_values(1):
            hoja.append_row(LOG_CSV_COLUMNAS)
        return hoja
    except Exception as e:
        logger.error(
            "No se pudo conectar a la hoja de log (%s). Se usará el CSV local.", e
        )
        return None


def _guardar_log_csv(fila: list[str]) -> None:
    """Log de respaldo en CSV local. Protegido con lock porque varios
    usuarios pueden escribir al mismo tiempo; no persiste entre redeploys."""
    with _log_lock:
        try:
            existe = os.path.isfile(LOG_FILE)
            with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not existe:
                    writer.writerow(LOG_CSV_COLUMNAS)
                writer.writerow(fila)
        except OSError as e:
            logger.error("No se pudo escribir el log en CSV: %s", e)


def guardar_log(mensaje: str, respuesta: str, modelo: str) -> None:
    """
    Guarda cada intercambio. Si hay una hoja de Google Sheets configurada
    se usa esa (persiste entre redeploys); si no, o si falla la escritura,
    cae de vuelta al CSV local. Un fallo al loguear nunca debe tumbar la
    conversación del usuario.

    `modelo` puede ser el nombre real del modelo de Gemini, o la etiqueta
    "instantánea (sin IA)" para las respuestas de los botones fijos, así
    el panel de métricas puede distinguir cuánto tráfico se resuelve sin
    costo de API.
    """
    fila = [datetime.now(timezone.utc).isoformat(), modelo, mensaje, respuesta]
    hoja = _obtener_hoja_log()
    if hoja is not None:
        try:
            hoja.append_row(fila)
            return
        except Exception as e:
            logger.error("Falló la escritura en Sheets (%s). Se usa CSV local.", e)
    _guardar_log_csv(fila)


def _leer_filas_log() -> list[dict[str, str]]:
    """Lee todo el historial de log disponible, para el panel de métricas."""
    hoja = _obtener_hoja_log()
    if hoja is not None:
        try:
            return hoja.get_all_records()
        except Exception as e:
            logger.warning("No se pudo leer la hoja de log (%s).", e)
    if os.path.isfile(LOG_FILE):
        try:
            with open(LOG_FILE, newline="", encoding="utf-8") as f:
                return list(csv.DictReader(f))
        except OSError:
            pass
    return []


# =============================================================================
# PANEL DE MÉTRICAS
# Nota: la contraseña es solo un filtro básico para un piloto de bajo
# riesgo (no hay límite de intentos, ni hash, ni sesión real). No reemplaza
# autenticación de verdad si esto se usa en un contexto más sensible.
# =============================================================================

def mostrar_panel_metricas() -> None:
    filas = _leer_filas_log()

    _catalogo_filas, origen_catalogo, motivo_fallback = cargar_catalogo()
    if origen_catalogo == "sheet":
        st.success(f"Catálogo activo: hoja de cálculo ({len(_catalogo_filas)} productos).")
    elif motivo_fallback:
        st.warning(
            f"Catálogo activo: embebido en el código. Hay una hoja "
            f"configurada pero no se pudo usar ({motivo_fallback})."
        )
    else:
        st.info("Catálogo activo: embebido en el código (no hay hoja configurada).")

    if not filas:
        st.info("Todavía no hay conversaciones registradas.")
        return

    total = len(filas)
    st.metric("Conversaciones registradas", total)

    instantaneas = sum(1 for f in filas if "instant" in (f.get("modelo") or "").lower())
    if instantaneas:
        st.caption(f"De esas, {instantaneas} se resolvieron con botones instantáneos (costo cero de API).")

    por_dia: Counter[str] = Counter()
    for f in filas:
        fecha = str(f.get("fecha_utc", ""))[:10]
        if fecha:
            por_dia[fecha] += 1
    if por_dia:
        st.caption("Conversaciones por día")
        st.bar_chart(dict(sorted(por_dia.items())))

    normalizadas: Counter[str] = Counter()
    for f in filas:
        pregunta = (f.get("mensaje_usuario") or "").strip().lower().rstrip("?.! ")
        if pregunta:
            normalizadas[pregunta] += 1
    if normalizadas:
        st.caption("Preguntas más frecuentes")
        for pregunta, veces in normalizadas.most_common(10):
            st.write(f"**{veces}×** — {pregunta}")

    derivadas = sum(
        1 for f in filas
        if "wa.me" in (f.get("respuesta_bot") or "")
        or "asesor" in (f.get("respuesta_bot") or "").lower()
    )
    porcentaje = (derivadas / total * 100) if total else 0
    st.metric("Respuestas que mencionan WhatsApp/asesor", f"{porcentaje:.0f}%")
    st.caption(
        "Estimado por palabras clave en la respuesta del bot (no es un dato "
        "exacto). Si el log vive solo en el CSV local (sin hoja de Sheets "
        "configurada), estos datos son solo desde el último reinicio de la app."
    )


# =============================================================================
# INTERFAZ
# =============================================================================

def _responder_instantanea(clave: str, etiqueta_boton: str) -> None:
    """Atiende un botón de respuesta instantánea: la agrega directamente al
    historial de la conversación, sin pasar por Gemini, y la loguea. No
    cuenta para MAX_MENSAJES_POR_SESION porque no usa la API."""
    texto = CANNED_ANSWERS[clave]
    st.session_state.messages.append({"role": "user", "content": etiqueta_boton})
    st.session_state.messages.append({"role": "assistant", "content": texto})
    guardar_log(etiqueta_boton, texto, "instantánea (sin IA)")


def main() -> None:
    st.set_page_config(page_title="MacBot", page_icon="🍎")
    st.title("🍎 MacBot - Asistente Virtual")
    st.caption("Pregunta por productos Apple, garantías, envíos y más.")

    # Se inicializa ANTES del sidebar porque algunos botones (respuestas
    # instantáneas) escriben directamente en st.session_state.messages.
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "turnos" not in st.session_state:
        st.session_state.turnos = 0

    cliente = obtener_cliente()
    try:
        modelo = resolver_modelo(cliente)
    except RuntimeError as e:
        logger.error(str(e))
        st.error("El asistente no está disponible en este momento. Intenta más tarde.")
        st.stop()

    chat = obtener_chat_de_sesion(cliente, modelo)

    with st.sidebar:
        st.subheader("Preguntas rápidas")
        st.caption(f"Modelo activo: {modelo.nombre}")
        _, origen_catalogo, _motivo = cargar_catalogo()
        st.caption(
            "Catálogo: hoja de cálculo" if origen_catalogo == "sheet"
            else "Catálogo: embebido en el código"
        )

        for accion in QUICK_ACTIONS:
            if st.button(accion.etiqueta, width="stretch", key=f"quick_{accion.tipo}_{accion.valor}"):
                if accion.tipo == "instantanea":
                    _responder_instantanea(accion.valor, accion.etiqueta)
                else:
                    st.session_state.pending_prompt = accion.valor

        st.link_button(
            "💬 Hablar con un asesor (WhatsApp)",
            INFO_TIENDA.whatsapp_link,
            width="stretch",
        )

        with st.expander("📋 Ver catálogo completo"):
            filas_catalogo, _origen, _motivo = cargar_catalogo()
            st.dataframe(
                [
                    {
                        "Categoría": f["categoria"],
                        "Producto": f["producto"],
                        "Detalle": f["detalle"],
                        "Precio": f["precio_texto"],
                    }
                    for f in filas_catalogo
                ],
                hide_index=True,
                width="stretch",
            )

        st.divider()
        if st.button(
            "🔄 Nueva conversación",
            width="stretch",
            help="Borra el historial visible y empieza un chat nuevo. El límite de mensajes por sesión no se reinicia.",
        ):
            st.session_state.messages = []
            st.session_state.pop("chat", None)
            st.session_state.pop("pending_prompt", None)
            st.rerun()

        st.divider()
        with st.expander("📊 Métricas (admin)"):
            clave_admin = _leer_secreto("ADMIN_PASSWORD")
            if not clave_admin:
                st.caption(
                    "No configurado. Agrega ADMIN_PASSWORD en Secrets para "
                    "habilitar este panel."
                )
            else:
                intento = st.text_input("Contraseña", type="password", key="clave_admin_input")
                if intento and intento == clave_admin:
                    mostrar_panel_metricas()
                elif intento:
                    st.error("Contraseña incorrecta.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Escribe tu pregunta...")
    if not prompt and st.session_state.get("pending_prompt"):
        prompt = st.session_state.pop("pending_prompt")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if st.session_state.turnos >= MAX_MENSAJES_POR_SESION:
                texto = (
                    "Llegamos al límite de mensajes para esta conversación 🙏. "
                    f"Para seguir, escríbenos directo por WhatsApp: {INFO_TIENDA.whatsapp_link}"
                )
                st.markdown(texto)
            else:
                with st.spinner("Escribiendo..."):
                    texto = enviar_con_reintentos(chat, prompt)
                st.markdown(texto)
                st.session_state.turnos += 1

        st.session_state.messages.append({"role": "assistant", "content": texto})
        guardar_log(prompt, texto, modelo.nombre)


if __name__ == "__main__":
    main()
