import streamlit as st
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
import csv
import os
import time
from datetime import datetime

# =========================================================================
# CONFIGURACIÓN DEL CLIENTE
# =========================================================================
# En Streamlit Cloud la API key vive en st.secrets (Settings > Secrets).
# En local puedes usar una variable de entorno como respaldo.
api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
if not api_key:
    st.error(
        "Falta la API key de Gemini. Agrégala en Settings > Secrets "
        "como GEMINI_API_KEY = \"tu_clave\"."
    )
    st.stop()

client = genai.Client(api_key=api_key)

# Google retira y renombra modelos de Gemini con frecuencia. En vez de
# depender de un solo nombre fijo, probamos una lista de candidatos de
# más a menos nuevo y nos quedamos con el primero que responda.
CANDIDATE_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]

# =========================================================================
# CONOCIMIENTO DEL NEGOCIO
# =========================================================================
system_prompt = """
Eres "MacBot", el asistente virtual de MacStore Ejemplo, una tienda
especializada en productos Apple. Respondes de forma amable, clara y
breve (máximo 3 párrafos), con emojis moderados.

## Datos de la tienda
- Horario: Lunes a sábado 9:00-20:00, domingo 10:00-18:00 (festivos: 10:00-17:00).
- Dirección: Centro Comercial X, local 123, Bogotá.
- WhatsApp: +57 300 000 0000 → https://wa.me/573000000000
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
    system_instruction=system_prompt,
    temperature=0.4,
    max_output_tokens=500,
)

LOG_FILE = "logs.csv"


def guardar_log(mensaje, respuesta, modelo):
    """Guarda cada intercambio en un CSV local (se reinicia si la app se redeploya)."""
    existe = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not existe:
            writer.writerow(["fecha", "modelo", "mensaje_usuario", "respuesta_bot"])
        writer.writerow([datetime.now().isoformat(), modelo, mensaje, respuesta])


@st.cache_resource(show_spinner=False)
def crear_chat():
    """
    Prueba los modelos candidatos en orden y crea la sesión de chat con el
    primero que responda. Se cachea para toda la vida del proceso, así no
    repetimos la prueba en cada mensaje.
    """
    ultimo_error = None
    for nombre_modelo in CANDIDATE_MODELS:
        try:
            chat = client.chats.create(model=nombre_modelo, config=GENERATION_CONFIG)
            # Ping corto para confirmar que el modelo de verdad responde
            chat.send_message("Hola")
            return chat, nombre_modelo
        except genai_errors.APIError as e:
            ultimo_error = e
            continue
    raise RuntimeError(
        f"Ninguno de los modelos candidatos respondió. Último error: {ultimo_error}"
    )


def enviar_con_reintentos(chat, mensaje, intentos=3):
    """Envía un mensaje con reintento y backoff exponencial ante errores 429/5xx."""
    espera = 1
    for intento in range(intentos):
        try:
            respuesta = chat.send_message(mensaje)
            return respuesta.text
        except genai_errors.APIError as e:
            codigo = getattr(e, "code", None)
            if codigo in (429, 500, 502, 503) and intento < intentos - 1:
                time.sleep(espera)
                espera *= 2
                continue
            return f"Ocurrió un error: {e}. Intenta de nuevo en unos segundos."
    return "No pude obtener respuesta en este momento. Intenta de nuevo en unos segundos."


# =========================================================================
# INTERFAZ
# =========================================================================
st.set_page_config(page_title="MacBot", page_icon="🍎")
st.title("🍎 MacBot - Asistente Virtual")
st.caption("Pregunta por productos Apple, garantías, envíos y más.")

try:
    chat, modelo_activo = crear_chat()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

with st.sidebar:
    st.subheader("Preguntas rápidas")
    st.caption(f"Modelo activo: {modelo_activo}")
    preguntas_rapidas = [
        "¿Cuál es el horario de la tienda?",
        "¿Qué iPhone me recomiendas para fotografía?",
        "¿Cómo funciona la garantía?",
        "Quiero hablar con un asesor",
    ]
    for pregunta in preguntas_rapidas:
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
    guardar_log(prompt, texto, modelo_activo)
