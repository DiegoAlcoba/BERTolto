## Detección automatizada de vulnerabilidades
### Estructura y organización de directorios

.

├── caddy

│   └── Caddyfile

├── data

│   ├── db\_init

│   │   └── init.sql

│   ├── db\_storage

│   ├── n8n\_storage

│   ├── nocodb\_data

│   ├── redisinsight\_storage

│   └── redis\_storage

├── docker-compose.yml

├── model\_api

│   ├── app.py

│   ├── Dockerfile

│   └── infer.py

├── models

│   └── bertolto\_v1

│       ├── config.json

│       ├── eval\_comment\_level.json

│       ├── inference\_meta.json

│       ├── merges.txt

│       ├── model.safetensors

│       ├── special\_tokens\_map.json

│       ├── threshold.json

│       ├── tokenizer\_config.json

│       ├── tokenizer.json

│       ├── val\_pooled_labels.npy

│       ├── val\_pooleds\_cores.npy

│       └── vocab.json

└── README.md


Los archivos no versionados en el repositorio es debido a que se generan con la construcción de la imagen Docker, contienen credenciales, o son demasiado pesados para hacerlo, como por ejemplo el directorio "models".

### caddy
Configuración reverse-proxy para el acceso a los servicios desde cualquier navegador sin necesidad de VPNs o configuraciones complejas.

### data
Directorios de almacenamiento de los datos de configuración de la base de datos (db\_init) e información de los servicios.

### docker-compose.yml
Archivo de configuración del contenedor compose de Docker utilizado para desplegar todos los servicios de forma sencilla y aislada.

### model\_api
API del modelo de inferencia para el procesamiento de los comentarios obtenidos de GitHub.

### models/bertolto\_v1
Directorio en el que ubicar los modelos de inferencia que se quiera utilizar, ya sea nuevas versiones en un futuro u otros modelos que se deseen probar y comparar rendimiento.
