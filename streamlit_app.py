import streamlit as st
import pandas as pd
import folium
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
from folium import plugins
from datetime import datetime

def mostrar_tabla_completa(archivo_csv):
    try:
        df = pd.read_csv(archivo_csv, sep=";")
        df = df.iloc[:, 1:]
        st.dataframe(df)
    except FileNotFoundError:
        st.error(f"No se encontró el archivo: {archivo_csv}")

def contar_sismos_entre_fechas(df, fecha_inicio, fecha_fin):
    df['FECHA_UTC'] = pd.to_datetime(df['FECHA_UTC'], format='%Y%m%d')
    fecha_inicio = pd.to_datetime(fecha_inicio)
    fecha_fin = pd.to_datetime(fecha_fin) + pd.DateOffset(days=1)
    mask = (df['FECHA_UTC'] >= fecha_inicio) & (df['FECHA_UTC'] <= fecha_fin)
    total_sismos = df.loc[mask].shape[0]
    return total_sismos

def calcular_profundidad_promedio(df, fecha_inicio, fecha_fin):
    df['FECHA_UTC'] = pd.to_datetime(df['FECHA_UTC'], format='%Y%m%d')
    fecha_inicio = pd.to_datetime(fecha_inicio)
    fecha_fin = pd.to_datetime(fecha_fin) + pd.DateOffset(days=1)
    mask = (df['FECHA_UTC'] >= fecha_inicio) & (df['FECHA_UTC'] <= fecha_fin)
    df_fechas = df.loc[mask]
    profundidades = df_fechas['PROFUNDIDAD']
    profundidad_promedio = profundidades.mean()
    return profundidad_promedio

def calcular_magnitud_promedio(df, fecha_inicio, fecha_fin):
    df['FECHA_UTC'] = pd.to_datetime(df['FECHA_UTC'], format='%Y%m%d')
    fecha_inicio = pd.to_datetime(fecha_inicio)
    fecha_fin = pd.to_datetime(fecha_fin) + pd.DateOffset(days=1)
    mask = (df['FECHA_UTC'] >= fecha_inicio) & (df['FECHA_UTC'] <= fecha_fin)
    df_fechas = df.loc[mask]
    magnitudes = df_fechas['MAGNITUD']
    magnitud_promedio = magnitudes.mean()
    return magnitud_promedio

def generar_mapa_sismos(df, fecha_inicio, fecha_fin, latitud_centro, longitud_centro, zoom_inicial, num_puntos):
    fecha_inicio = np.datetime64(fecha_inicio)  # Convertir a datetime64[ns]
    fecha_fin = np.datetime64(fecha_fin)  # Convertir a datetime64[ns]
    df_fechas = df[(df['FECHA_UTC'] >= fecha_inicio) & (df['FECHA_UTC'] <= fecha_fin)]

    if len(df_fechas) > num_puntos:
        df_fechas = df_fechas.sample(num_puntos)  # Muestra aleatoria de num_puntos

    mapa = folium.Map(location=[latitud_centro, longitud_centro], zoom_start=zoom_inicial)

    # Crear un grupo de marcadores con etiquetas emergentes
    marker_cluster = MarkerCluster().add_to(mapa)

    # Generar una secuencia de colores entre verde y rojo
    colormap = plt.get_cmap('RdYlGn')  # RdYlGn es una paleta de colores verde-rojo
    magnitudes = df_fechas['MAGNITUD']
    min_magnitud = magnitudes.min()
    max_magnitud = magnitudes.max()

    for index, row in df_fechas.iterrows():
        lat = row['LATITUD']
        lon = row['LONGITUD']
        magnitud = row['MAGNITUD']
        fecha = row['FECHA_UTC']
        hora = datetime.strptime(str(row['HORA_UTC']), "%H%M%S").strftime('%H:%M:%S')  # Convertir la hora a formato HH:MM:SS
        profundidad = row['PROFUNDIDAD']

        # Normalizar la magnitud entre 0 y 1
        normalized_magnitude = (magnitud - min_magnitud) / (max_magnitud - min_magnitud)

        # Obtener el color correspondiente a la magnitud normalizada
        color = mpl_colors.rgb2hex(colormap(normalized_magnitude))

        # Verificar si el color es válido, de lo contrario, utilizar 'white' como valor predeterminado
        if color not in ['white', 'blue', 'pink', 'red', 'lightred', 'green', 'lightgray', 'black', 'cadetblue',
                         'gray', 'purple', 'lightgreen', 'lightblue', 'beige', 'darkpurple', 'darkblue',
                         'darkred', 'orange', 'darkgreen']:
            color = 'white'

        # Crear una etiqueta emergente con la información del sismo
        html = f"<b>Fecha:</b> {fecha}<br>"
        html += f"<b>Hora:</b> {hora}<br>"
        html += f"<b>Latitud:</b> {lat}<br>"
        html += f"<b>Longitud:</b> {lon}<br>"
        html += f"<b>Magnitud:</b> {magnitud} ML<br>"
        html += f"<b>Profundidad:</b> {profundidad} Km<br>"

        # Agregar el marcador al grupo con la etiqueta emergente
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(html, max_width=250),
            icon=folium.Icon(color=color)
        ).add_to(marker_cluster)

    # Añadir el mapa oscuro después de crear el objeto de mapa
    folium.TileLayer(tiles='CartoDB dark_matter', control=False).add_to(mapa)

    return mapa

def grafico_sismos_magnitud(archivo_csv):
    # Cargar el archivo CSV
    df = pd.read_csv(archivo_csv, delimiter=';')

    # Obtener los valores de magnitud y contar la frecuencia
    magnitudes = df['MAGNITUD']
    magnitudes = magnitudes.value_counts().sort_index().reset_index()
    magnitudes.columns = ['Magnitud (ML)', 'Número de sismos']

    # Crear el gráfico de barras con Altair
    chart = alt.Chart(magnitudes).mark_bar(color='#CD4040').encode(
        x='Magnitud (ML):O',
        y='Número de sismos:Q',
        tooltip=['Magnitud (ML)', 'Número de sismos']
    ).properties(
        width=600,
        height=400,
        title='Número de sismos según la magnitud'
    )

    # Mostrar el gráfico en Streamlit
    st.altair_chart(chart)

def grafico_sismos_profundidad(archivo_csv):
    # Leer el archivo CSV
    df = pd.read_csv(archivo_csv, delimiter=';')

    # Obtener los valores de profundidad y contar la frecuencia
    profundidades = df['PROFUNDIDAD']
    profundidades = profundidades.value_counts().reset_index()
    profundidades.columns = ['Profundidad (Km)', 'Número de sismos']

    # Crear el gráfico de puntos unidos por líneas con Altair
    chart = alt.Chart(profundidades).mark_line(color='#BA43AF').encode(
        x='Profundidad (Km):Q',
        y='Número de sismos:Q'
    ).properties(
        width=600,
        height=400,
        title='Número de sismos según la profundidad'
    )

    # Mostrar el gráfico en Streamlit
    st.altair_chart(chart)

def grafico_promedio_profundidad_por_anio(archivo_csv):
    # Leer el archivo CSV
    df = pd.read_csv(archivo_csv, delimiter=';')

    # Convertir la columna 'FECHA_UTC' a tipo datetime
    df['FECHA_UTC'] = pd.to_datetime(df['FECHA_UTC'], format='%Y%m%d')

    # Obtener el año de cada fecha y calcular el promedio de profundidad por año
    df['AÑO'] = df['FECHA_UTC'].dt.year
    promedio_profundidad_por_anio = df.groupby('AÑO')['PROFUNDIDAD'].mean().reset_index()

    # Crear el gráfico de barras con Altair
    chart = alt.Chart(promedio_profundidad_por_anio).mark_bar(color='#E58225').encode(
        x='AÑO:N',
        y='PROFUNDIDAD:Q',
        tooltip=['AÑO', 'PROFUNDIDAD']
    ).properties(
        width=600,
        height=400,
        title='Promedio de profundidad de sismos por año'
    )

    # Mostrar el gráfico en Streamlit
    st.altair_chart(chart)

def crear_grafico_dispersion(archivo_csv):
    # Cargar los datos desde el archivo CSV
    data = pd.read_csv(archivo_csv, sep=';')

    # Crear el gráfico de dispersión
    scatter_plot = alt.Chart(data).mark_circle(size=60).encode(
        x='MAGNITUD',
        y='PROFUNDIDAD',
        color='MAGNITUD',
        tooltip=['ID', 'FECHA_UTC', 'HORA_UTC', 'LATITUD', 'LONGITUD', 'PROFUNDIDAD', 'MAGNITUD', 'FECHA_CORTE']
    ).properties(
        width=600,
        height=400,
        title='Magnitud vs. Profundidad de Sismos'
    )

    # Mostrar el gráfico utilizando Streamlit
    st.altair_chart(scatter_plot, use_container_width=True)

def grafico_lineas_magnitud_promedio_por_anio(archivo_csv):
    # Cargar los datos desde el archivo CSV
    data = pd.read_csv(archivo_csv, sep=';')

    # Convertir la columna FECHA_UTC a tipo fecha
    data['FECHA_UTC'] = pd.to_datetime(data['FECHA_UTC'], format='%Y%m%d')

    # Extraer el año de la fecha
    data['Año'] = data['FECHA_UTC'].dt.year

    # Calcular la magnitud promedio por año
    magnitud_promedio_por_anio = data.groupby('Año')['MAGNITUD'].mean().reset_index(name='Magnitud Promedio')

    # Crear el gráfico de líneas
    line_chart = alt.Chart(magnitud_promedio_por_anio).mark_line(color="#65BF43").encode(
        x=alt.X('Año:O', title='Año'),
        y=alt.Y('Magnitud Promedio:Q', title='Magnitud Promedio'),
        tooltip=['Año:O', 'Magnitud Promedio:Q']
    ).properties(
        width=600,
        height=400,
        title='Evolución de la Magnitud Promedio por Año'
    )

    # Mostrar el gráfico utilizando Streamlit
    st.altair_chart(line_chart, use_container_width=True)

def grafico_barras_promedio_magnitud_profundidad(archivo_csv):
    # Leer el archivo CSV
    data = pd.read_csv(archivo_csv, sep=';')

    # Convertir la columna 'FECHA_UTC' a tipo string
    data['FECHA_UTC'] = data['FECHA_UTC'].astype(str)

    # Obtener los años únicos en el dataset
    years = data['FECHA_UTC'].str[:4].unique()

    # Solicitar al usuario seleccionar dos años
    selected_years = st.multiselect('Selecciona dos años:', years)

    if len(selected_years) != 2:
        st.warning('Por favor, selecciona exactamente dos años.')
        return

    # Filtrar los datos por los años seleccionados
    filtered_data = data[data['FECHA_UTC'].str[:4].isin(selected_years)]

    # Calcular los promedios de magnitud y profundidad por año
    average_values = filtered_data.groupby([filtered_data['FECHA_UTC'].str[:4]])[['MAGNITUD', 'PROFUNDIDAD']].mean().reset_index()

    # Crear los gráficos de barras de magnitud y profundidad
    chart_magnitud = alt.Chart(average_values).mark_bar(color='steelblue').encode(
        alt.X('FECHA_UTC:N', title='Año'),
        alt.Y('MAGNITUD:Q', title='Promedio de Magnitud')
    ).properties(
        width=400,
        height=300
    )

    chart_profundidad = alt.Chart(average_values).mark_bar(color='orange').encode(
        alt.X('FECHA_UTC:N', title='Año'),
        alt.Y('PROFUNDIDAD:Q', title='Promedio de Profundidad')
    ).properties(
        width=400,
        height=300
    )

    # Combinar los gráficos horizontalmente
    chart_combined = alt.hconcat(chart_magnitud, chart_profundidad)

    # Mostrar el gráfico combinado
    st.altair_chart(chart_combined)

latitud_centro = -9.189967
longitud_centro = -75.015152
zoom_inicial = 5
num_puntos = 100

st.set_page_config(layout="wide")

# Cargar el archivo CSV
archivo_csv = 'https://raw.githubusercontent.com/Arianna-cuellar/-catalogo-sismico-Dashboard-Interativo/main/Catalogo1960_2021.csv'
df = pd.read_csv(archivo_csv, delimiter=';')

# Crear la aplicación Streamlit
st.title("Catálogo sísmico IGP")

# Establecer límites de fecha
fecha_inicio_min = pd.to_datetime('1960-01-13').date()
fecha_fin_max = pd.to_datetime('2021-12-31').date()

# Botones en el sidebar
opciones = ['Inicio', 'Promedios']
opcion = st.sidebar.radio('Seleccione una opción:', opciones)

# Interfaz de inicio
if opcion == 'Inicio':
    # Solicitar al usuario ingresar las fechas
    with st.sidebar:
        st.sidebar.write(
        """
        <h1 style="font-size: 30px; font-family: Century Gothic; color: #FFFFFF;">
        Margen de fechas
        </h1>
        <br>
        <style>label[for='fecha_inicio'], label[for='fecha_fin'] { margin-top: 10px; }</style>
        """
        ,
        unsafe_allow_html=True
    )
        fecha_inicio = st.date_input("Fecha de inicio", min_value=fecha_inicio_min, max_value=fecha_fin_max, value=fecha_inicio_min, key='fecha_inicio')
        fecha_fin = st.date_input("Fecha de fin", min_value=fecha_inicio_min, max_value=fecha_fin_max, value=fecha_fin_max, key='fecha_fin')
    mostrar_tabla_completa(archivo_csv)

    # Crea dos columnas utilizando contenedores div
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    # Verificar si se han ingresado las fechas
    if fecha_inicio and fecha_fin:
        if fecha_inicio > fecha_fin:
            st.error("La fecha de inicio debe ser anterior a la fecha de fin.")
        else:
            # Contar los sismos entre las fechas ingresadas
            total = contar_sismos_entre_fechas(df, fecha_inicio, fecha_fin)
            with col1:
                st.markdown(f"<div style='background-color:#222222; padding:10px; border-radius:10px; display: inline-block; width: 182px;'>\
                <h2 style='color:white; text-align:center; font-size: 20px; font-family:Century Gothic;'>N° de sismos</h2>\
                <h1 style='color:#5BD1F7; text-align:center; line-height: 0.3; font-family:Century Gothic'>{total}</h1>\
                <h2 style='color:white; text-align:center; font-size: 20px; font-family:Century Gothic;'>Entre las fechas</h3>\
                </div>", unsafe_allow_html=True)
        
                st.markdown("<br style='margin: -100px;'>", unsafe_allow_html=True)

                # Calcular la profundidad promedio entre las fechas ingresadas
                promedio_profundidad = calcular_profundidad_promedio(df, fecha_inicio, fecha_fin)
                st.markdown(f"<div style='background-color:#222222; padding:10px; border-radius:10px; display: inline-block; width: 182px;'>\
                    <h2 style='color:white; text-align:center; font-size: 12px; font-family:Century Gothic;'>Profundidad promedio</h2>\
                    <h1 style='color:#5BD1F7; text-align:center; line-height: 0.3; font-family:Century Gothic'>{promedio_profundidad:.2f}</h1>\
                    <h2 style='color:white; text-align:center; font-size: 20px; font-family:Century Gothic;'>Km</h2>\
                    </div>", unsafe_allow_html=True)
        
                st.markdown("<br style='margin: -100px;'>", unsafe_allow_html=True)

            # Calcular el promedio de la magnitud entre las fechas ingresadas
                promedio_magnitud = calcular_magnitud_promedio(df, fecha_inicio, fecha_fin)
                st.markdown(f"<div style='background-color:#222222; padding:10px; border-radius:10px; display: inline-block; width: 182px;'>\
                    <h2 style='color:white; text-align:center; font-size: 12px; font-family:Century Gothic;'>Magnitud promedio</h2>\
                    <h1 style='color:#5BD1F7; text-align:center; line-height: 0.3; font-family:Century Gothic'>{promedio_magnitud:.2f}</h1>\
                    <h2 style='color:white; text-align:center; font-size: 20px; font-family:Century Gothic;'>ML</h2>\
                    </div>", unsafe_allow_html=True)

                st.markdown("<br style='margin: -100px;'>", unsafe_allow_html=True)
            with col2:
            # Generar el mapa de sismos
                mapa = generar_mapa_sismos(df, fecha_inicio, fecha_fin, latitud_centro, longitud_centro, zoom_inicial, num_puntos)
                folium_static(mapa)

    grafico_sismos_magnitud(archivo_csv)
    grafico_sismos_profundidad(archivo_csv)
# Interfaz de promedios
elif opcion == 'Promedios':
    grafico_promedio_profundidad_por_anio(archivo_csv)
    crear_grafico_dispersion(archivo_csv)
    grafico_lineas_magnitud_promedio_por_anio(archivo_csv)
    grafico_barras_promedio_magnitud_profundidad(archivo_csv)
