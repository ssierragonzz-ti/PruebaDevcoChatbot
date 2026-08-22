"""
MacBot - Asistente virtual para una tienda de productos Apple.

Arquitectura:
- El cliente HTTP hacia la API de Gemini es un recurso compartido (sin
  estado propio de conversación) cacheado a nivel de proceso.
- Cada visitante tiene su propia sesión de chat, guardada en
  st.session_state, para no mezclar conversaciones entre usuarios.
- El modelo activo se resuelve una sola vez contra una lista de
  candidatos, para tolerar que Google retire o renombre modelos.
"""

from __future__ import annotations

import csv
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

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
CANDIDATE_MODELS: tuple[str, ...] = (
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
)

MAX_RETRIES = 3
RETRY_BASE_DELAY_SECONDS = 1.0
RETRYABLE_STATUS_CODES = {429, 500, 502, 503}

LOG_FILE = os.getenv("MACBOT_LOG_FILE", "logs.csv")
_log_lock = threading.Lock()  # protege escrituras concurrentes de varios usuarios

SYSTEM_PROMPT = """
Eres "MacBot", el asistente virtual de MacStore Ejemplo, una tienda
especializada en productos Apple. Respondes de forma amable, clara y
breve (máximo 3 párrafos), con emojis moderados.

## Datos de la tienda
- Horario: Lunes a sábado 9:00-20:00, domingo 10:00-18:00 (festivos: 10:00-17:00).
- Dirección: Centro Comercial X, local 123, Bogotá.
- WhatsApp: +57 300 000 0000 -> https://wa.me/573000000000
- Web: www.macstore-ejemplo.com
- Redes: @macstoreejemplo en Instagram

## Catálogo destacado
iPhone:
- iPhone 15: chip A16, cámara 48 MP, USB-C. Desde $4,299,000.
- iPhone 15 Pro: chip A17 Pro, titanio, teleobjetivo 48 MP. Desde $5,499,000.
- iPhone 16: chip A18, cámara Fusion, botón Acción y Control de Cámara. Desde $4,799,000.
- iPhone 16 Pro: chip A18 Pro, pantalla más grande, grabación espacial. Desde $5,999,000.

Mac:
- MacBook Air M3: 13.6", 8 GB RAM, 256 GB SSD. Desde $5,999,000.
- MacBook Pro M3 Pro: 14", 18 GB RAM, 512 GB SSD. Desde $11,999,000.
- Mac mini M4: compacto, ideal para escritorio con monitor propio. Desde $3,999,000.
- iMac M4: 24", todo en uno, colores vivos. Desde $6,999,000.
- Mac Studio: estaciones de trabajo de alto rendimiento (edición de video, 3D). Desde $18,999,000.

iPad:
- iPad (básico): pantalla 10.9", ideal para uso diario y estudio. Desde $2,199,000.
- iPad Air: chip M2, más potencia para diseño e ilustración. Desde $3,399,000.
- iPad Pro: pantalla OLED, con Apple Pencil Pro. Desde $6,299,000.
- iPad mini: portátil, buena para lectura y notas rápidas. Desde $2,899,000.

Accesorios y otros:
- AirPods Pro 2: cancelación activa de ruido, chip H2. $1,299,000.
- AirPods 4: nuevos, con o sin cancelación de ruido. Desde $799,000.
- Apple Watch Series 9: 41 mm, GPS. $2,399,000.
- Apple Watch Ultra 2: resistente, GPS + Cellular, para deporte extremo. $6,999,000.
- Apple TV 4K: streaming, ideal para complementar un Mac o iPhone. $1,499,000.
- HomePod mini: altavoz inteligente, se integra con el resto del ecosistema. $599,000.
- AirTag: localizador de objetos. $149,000 (pack x4: $499,000).
- Fundas, cargadores y cables originales: precios desde $69,000.

## Recomendaciones según necesidad
- Fotografía / contenido: iPhone 16 Pro o iPhone 15 Pro.
- Diseño gráfico / edición de video: MacBook Pro o Mac Studio.
- Estudio, oficina, uso general: MacBook Air o iPad Air.
- Toma de notas y lectura: iPad o iPad mini con Apple Pencil.
- Deporte y salud: Apple Watch (Ultra 2 si es deporte de alto rendimiento).
- Hogar inteligente: HomePod mini + Apple TV 4K.
- Evitar perder cosas: AirTag.

## Políticas
- Garantía: 1 año con Apple en todos los productos; AppleCare+ opcional (cobertura extendida y daños accidentales).
- Devoluciones: 30 días en empaque original con factura.
- Reparaciones: diagnóstico gratuito, solo repuestos originales, en centro de servicio autorizado.
- Envíos: Bogotá mismo día (pedidos antes de las 3pm), otras ciudades 1-2 días hábiles.
- Financiación: hasta 24 cuotas con tarjetas de crédito aliadas; consulta bancos participantes por WhatsApp.
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
"""

GENERATION_CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    temperature=0.4,
    max_output_tokens=500,
)

QUICK_QUESTIONS: tuple[str, ...] = (
    "¿Cuál es el horario de la tienda?",
    "¿Qué iPhone me recomiendas para fotografía?",
    "¿Cómo funciona la garantía?",
    "Quiero hablar con un asesor",
)


# =============================================================================
# TIPOS
# =============================================================================

@dataclass(frozen=True)
class ModeloResuelto:
    nombre: str


# =============================================================================
# CAPA DE INFRAESTRUCTURA (cacheada a nivel de proceso, sin estado de sesión)
# =============================================================================

def _leer_api_key() -> str:
    api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
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
    sin estado. Se cachea el nombre del primero que responda; no se
    cachea ninguna conversación aquí.
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
        except genai_errors.APIError as e:
            logger.warning("Modelo %s no disponible (%s), probando el siguiente.", nombre, e)
            ultimo_error = e
            continue

    raise RuntimeError(
        f"Ninguno de los modelos candidatos respondió. Último error: {ultimo_error}"
    )


# =============================================================================
# CAPA DE SESIÓN (una conversación por visitante)
# =============================================================================

def obtener_chat_de_sesion(cliente: genai.Client, modelo: ModeloResuelto):
    """Crea (o reutiliza) la sesión de chat propia de este usuario."""
    if "chat" not in st.session_state:
        st.session_state.chat = cliente.chats.create(model=modelo.nombre, config=GENERATION_CONFIG)
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
    """Envía un mensaje con reintento y backoff exponencial ante errores 429/5xx."""
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
    return "No pude obtener respuesta en este momento. Intenta de nuevo en unos segundos."


def guardar_log(mensaje: str, respuesta: str, modelo: str) -> None:
    """
    Guarda cada intercambio en un CSV local. Protegido con un lock porque
    varios usuarios pueden escribir al mismo tiempo; el archivo se reinicia
    si la app se redeploya (almacenamiento no persistente).
    """
    fila = [datetime.now(timezone.utc).isoformat(), modelo, mensaje, respuesta]
    with _log_lock:
        try:
            existe = os.path.isfile(LOG_FILE)
            with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not existe:
                    writer.writerow(["fecha_utc", "modelo", "mensaje_usuario", "respuesta_bot"])
                writer.writerow(fila)
        except OSError as e:
            # Un fallo al loguear no debe tumbar la conversación del usuario.
            logger.error("No se pudo escribir el log: %s", e)


# =============================================================================
# INTERFAZ
# =============================================================================

def main() -> None:
    st.set_page_config(page_title="MacBot", page_icon="🍎")
    st.title("🍎 MacBot - Asistente Virtual")
    st.caption("Pregunta por productos Apple, garantías, envíos y más.")

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
        for pregunta in QUICK_QUESTIONS:
            if st.button(pregunta, use_container_width=True):
                st.session_state.pending_prompt = pregunta

    if "messages" not in st.session_state:
        st.session_state.messages = []

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
            with st.spinner("Escribiendo..."):
                texto = enviar_con_reintentos(chat, prompt)
            st.markdown(texto)

        st.session_state.messages.append({"role": "assistant", "content": texto})
        guardar_log(prompt, texto, modelo.nombre)


if __name__ == "__main__":
    main()
