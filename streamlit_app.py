import streamlit as st
import google.generativeai as genai
import csv
import os
from datetime import datetime

# --- Configuración del modelo ---
# En Streamlit Cloud la API key se lee de st.secrets (Settings > Secrets).
# En local puedes usar una variable de entorno como respaldo.
api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
if not api_key:
    st.error(
        "Falta la API key de Gemini. Agrégala en Settings > Secrets "
        "como GEMINI_API_KEY = \"tu_clave\"."
    )
    st.stop()

genai.configure(api_key=api_key)

system_prompt = """
Eres un asistente virtual de una tienda especializada en productos Apple.
Tu nombre es "MacBot".
Responde de forma amable, clara y breve (máximo 3 párrafos).
Usa emojis moderados para dar calidez.

Información de la tienda:
- Nombre: MacStore Ejemplo
- Horario: Lunes a sábado 9:00-20:00, domingo 10:00-18:00.
- Dirección: Centro Comercial X, local 123.
- WhatsApp: +57 300 000 0000
- Web: www.macstore-ejemplo.com

Políticas:
- Garantía: 1 año con Apple, AppleCare+ opcional.
- Devoluciones: 30 días en empaque original con factura.
- Reparaciones: diagnóstico gratuito, repuestos originales.
- Envíos: Bogotá mismo día, otras ciudades 1-2 días hábiles.

Productos destacados:
- iPhone 15: chip A16, cámara 48 MP, USB-C. Desde $4,299,000.
- iPhone 15 Pro: chip A17 Pro, titanio, cámara 48 MP con teleobjetivo. Desde $5,499,000.
- MacBook Air M3: 13.6", 8 GB RAM, 256 GB SSD. Desde $5,999,000.
- MacBook Pro M3 Pro: 14", 18 GB RAM, 512 GB SSD. Desde $11,999,000.
- AirPods Pro 2: cancelación activa de ruido, chip H2. $1,299,000.
- Apple Watch Series 9: 41 mm, GPS. $2,399,000.

Recomendaciones:
- Fotografía: iPhone 15 Pro.
- Diseño gráfico: MacBook Pro.
- Estudio/oficina: MacBook Air.
- Deporte: AirPods Pro.
- Salud y fitness: Apple Watch.

Reglas:
- Si te preguntan por disponibilidad o stock, responde que no tienes acceso al inventario en tiempo real y sugiere contactar a un asesor por WhatsApp.
- Si el cliente quiere comprar, ofrécele el enlace de WhatsApp: https://wa.me/573000000000
- Si es fuera de horario, pídele nombre y teléfono para que un asesor lo contacte al día siguiente.
- No inventes precios ni características. Si no sabes algo, dilo y deriva a un humano.
- No pidas datos sensibles como números de tarjeta o contraseñas.
"""

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash-lite",
    system_instruction=system_prompt,
)

LOG_FILE = "logs.csv"


def guardar_log(mensaje, respuesta):
    """Guarda cada intercambio en un CSV local (se reinicia si el Space/app se redeploya)."""
    existe = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not existe:
            writer.writerow(["fecha", "mensaje_usuario", "respuesta_bot"])
        writer.writerow([datetime.now().isoformat(), mensaje, respuesta])


st.set_page_config(page_title="MacBot", page_icon="🍎")
st.title("🍎 MacBot - Asistente Virtual")
st.caption("Pregunta por productos Apple, garantías, envíos y más.")

# --- Estado de la conversación ---
if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])
    st.session_state.messages = []

# Mostrar historial ya renderizado
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Entrada del usuario
if prompt := st.chat_input("Escribe tu pregunta..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Escribiendo..."):
            try:
                respuesta = st.session_state.chat.send_message(prompt)
                texto = respuesta.text
            except Exception as e:
                texto = f"Ocurrió un error: {str(e)}. Intenta de nuevo en unos segundos."
        st.markdown(texto)

    st.session_state.messages.append({"role": "assistant", "content": texto})
    guardar_log(prompt, texto)
