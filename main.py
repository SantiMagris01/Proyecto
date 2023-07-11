import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from fastapi import FastAPI


app = FastAPI()

df = pd.read_csv("movies_modificado.csv")

@app.get('/')
def read_root():
    return {'message' : 'API para consultar datos de Peliculas'}

@app.get('/peliculas_idioma/{idioma}')
def peliculas_idioma(idioma: str):
    df_filtrado = df[df['original_language'] == idioma]
    cantidad_pelicualas = len(df_filtrado)
    return {'Idioma' : idioma, 'Cantidad de peliculas' : cantidad_pelicualas}


@app.get('/peliculas_duracion/{pelicula}')

def peliculas_duracion(pelicula: str):
    pelicula_filtrada = df[df['title'] == pelicula]
    
    if not pelicula_filtrada.empty:
        duracion = pelicula_filtrada['runtime'].values[0]
        anio = pelicula_filtrada['release_year'].values[0]

        respuesta = {
            'Pelicula': pelicula,
            'Duración': duracion,
            'Año': anio
        }
    else:
        respuesta = {
            'Pelicula': pelicula,
            'Duración': None,
            'Año': None
        }

    return respuesta


@app.get('/franquicia/{franquicia}')
def franquicia(franquicia : str):
    franquicica_filtrada = df[df['collection_name'] == franquicia]
    cantidad = len(franquicica_filtrada)
    ganancia_total = franquicica_filtrada['revenue'].sum()
    ganacia_promedio = ganancia_total/cantidad

    return {'Franquicia' : franquicia, 'Cantidad' : cantidad, 'Ganancia Total' : ganancia_total, 'Ganancia Promedio' : ganacia_promedio}




@app.get('/peliculas_pais/{pais}')
def peliculas_pais(pais: str):
    
    pais_filtrado = df[df['production_countries_names'].apply(lambda x: pais in x)]
    cantidad = len(pais_filtrado)

    return {'Pais' : pais, 'Cantidad de Peliculas' : cantidad}


@app.get('/productoras_exitosas/{productora}')
def productoras_exitosas(productora : str):
    productora_filtrada = df[df['production_companies_names'].apply(lambda x: productora in x)]
    cantidad = len(productora_filtrada)
    revenue = productora_filtrada['revenue'].sum()

    respuesta = {
        'productora' : productora,
        'cantidad' : cantidad,
        'revenue' : revenue
    }

    return respuesta


@app.get('/get_director/{director}')
def get_director(director:str):
    director_filtrado = df[df['director_name'] == director]
    retorno_total_director = director_filtrado['return'].sum()
    peliculas = []

    for index, row in director_filtrado.iterrows():
        pelicula = {
            'nombre' : row['title'],
            'fecha de lanzamiento' : row['release_date'],
            'retorno individual' : row['return'],
            'costo': row['budget'],
            'ganancia': row['revenue']
        }
        peliculas.append(pelicula)
    return {'director' : director, 'Retorno total del director' : retorno_total_director}, peliculas


# Seleccionamos las columnas relevantes
selected_columns = ['overview', 'genres_names', 'cast_names', 'director_name', 'popularity', 'vote_average', 'production_companies_names', 'release_year']
df_selected = df[selected_columns]

# Completamos los valores faltantes
df_selected = df_selected.fillna('')

# Combinamos las columnas seleccionadas
df_selected['features'] = df_selected[selected_columns].apply(lambda x: ' '.join(str(val) for val in x), axis=1)

# Inicilizamos TF-IDF 
vectorizer = TfidfVectorizer()
features_matrix = vectorizer.fit_transform(df_selected['features'])

# Inicializamos KVecinos mas cercanos
knn_model = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='cosine')
knn_model.fit(features_matrix)

@app.get('/recomendaciones/{titulo}')
def recommend_movies(movie_title, top_n=5):
    # Encontramos las pelicula
    movie_index = df[df['title'] == movie_title].index[0]

    # Encontramos las peliculas similares
    _, indices = knn_model.kneighbors(features_matrix[movie_index])

    # Extraemos los titulos de las peliculas
    top_movies = df.loc[indices.flatten()[1:top_n+1], 'title'].tolist()

    return {'Pelicula': movie_title, 'Peliculas Recomendadas': top_movies}
