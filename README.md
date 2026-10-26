# VigIA: reconocimiento de comportamiento humano en cámaras de seguridad

Prototipo de software derivado de la revisión sistemática *Reconocimiento del comportamiento humano en cámaras
de seguridad mediante visión artificial y aprendizaje profundo (2021-2026)*. Analiza en tiempo real la webcam o
cámaras IP (RTSP), detecta comportamientos de riesgo y los muestra en un panel web con evidencia y explicación.

| Comportamiento | Cómo se detecta | Alerta |
|---|---|---|
| Violencia o pelea | Dos personas a menos de 1.2 alturas con muñecas, codos o tobillos moviéndose rápido respecto del torso (≥ 1.8 alturas/s durante ≥ 0.6 s) | Alta |
| Caída | El centro de la cadera baja ≥ 0.35 alturas en menos de 2.5 s y el tronco queda a más de 60° de la vertical | Alta |
| Persona en el suelo | Tronco horizontal durante ≥ 5 s | Alta |
| Intrusión | Los pies de una persona entran en una zona marcada como «restringida» | Alta |
| Merodeo | Permanencia mayor a 60 s en una zona de «merodeo» (o en la cámara si no hay zonas) | Media |
| Objeto abandonado | Mochila, bolso o maleta inmóvil ≥ 30 s sin nadie a menos de 2.5 alturas | Media |
| Posible hurto | Un objeto que estaba quieto y asociado a una persona se desplaza junto a otra persona | Alta |
| Expresión facial | Alegría, neutral, sorpresa, tristeza, enojo, miedo o desagrado, por persona | Informativa |

Todas las distancias se miden en «alturas de persona», así que las reglas no dependen de la resolución ni de la
distancia a la cámara. Todos los umbrales se cambian en `config.yaml` o desde **Ajustes** en el panel.

## Relación con la revisión sistemática

El diseño responde a los vacíos que encontró la revisión:

- **Tiempo real en CPU (PE4).** Se usan modelos *nano* (YOLO11n-pose y YOLO11n) y un único paso de pose por frame;
  el detector de objetos corre uno de cada tres frames. No hace falta GPU.
- **Explicabilidad (PE4, ética).** Cada alerta trae la medida que la disparó (por ejemplo, «extremidades a 3.1 alturas/s
  durante 0.8 s, con 0.6 alturas de separación») y una captura con la evidencia resaltada.
- **Privacidad por diseño (PE3).** Los rostros se pixelan antes de mostrarse o guardarse, el video no se graba, las
  capturas se borran a los 7 días y todo se procesa en el equipo local.
- **Evaluación reproducible (PE4).** `scripts/evaluar_rwf2000.py` mide accuracy, precisión, recall, F1 y FPS en RWF-2000,
  el mismo conjunto que usan los estudios revisados, y calibra el umbral de violencia.
- **El operador decide.** Las alertas son señales para una persona, no acciones automáticas.

## Requisitos

- Windows 10/11, Linux o macOS, con **Python 3.10 a 3.12** (python.org; marque «Add python.exe to PATH»).
- 8 GB de RAM. Funciona solo con CPU; una GPU NVIDIA lo acelera, pero no es necesaria.
- Conexión a internet solo la primera vez (dependencias y modelos, unos 600 MB en total).

## Instalación y uso en Windows

1. Descomprima la carpeta `vigia`.
2. **Modo demo** (sin cámara ni modelos): doble clic en `demo.bat`. Se abre el panel con una escena simulada de
   100 s donde aparecen todos los comportamientos. Sirve para probar y para exponer.
3. **Con la webcam:** doble clic en `iniciar.bat`. La primera vez crea un entorno de Python, instala las dependencias
   y descarga los modelos; luego abre `http://127.0.0.1:8000`.
4. **Cámara IP:** en el panel, **+ Cámara** y escriba la dirección RTSP. Ejemplos:
   - Hikvision: `rtsp://usuario:clave@192.168.1.64:554/Streaming/Channels/101`
   - Dahua: `rtsp://usuario:clave@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0`
   - Un video: la ruta a un `.mp4`, que se reproduce en bucle (útil para probar con clips de UCF-Crime o RWF-2000).
5. **Zonas:** botón **Zonas**, elija la cámara, haga clic sobre la imagen para marcar los vértices y doble clic para
   cerrar el polígono. Las «restringidas» generan intrusión; las de «merodeo», alerta por permanencia.

En Linux o macOS: `./iniciar.sh` o `./iniciar.sh --demo`.

Comandos manuales, si prefiere no usar los `.bat`:

```bash
python -m venv .venv && .venv\Scripts\activate      # en Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python scripts/descargar_modelos.py
python -m vigia            # o: python -m vigia --demo
```

## Evaluación con RWF-2000

RWF-2000 tiene 2 000 clips de 5 s de cámaras de seguridad reales, con y sin pelea. Se solicita a sus autores en
[github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection).

```bash
python scripts/evaluar_rwf2000.py --datos D:/RWF-2000/val --max-videos 100
```

El script calcula la pose de cada clip una vez, prueba varios umbrales y muestra una tabla con accuracy, precisión,
recall, F1 y los FPS del modelo en su computadora; deja el detalle por video en `resultados_rwf2000.csv`. Use el mejor
umbral en `config.yaml` (`reglas.violencia.velocidad_extremidades`). Estas cifras son las que corresponde reportar
en el informe del producto: el umbral por defecto (1.8) es un valor de partida, no un resultado medido.

## Pruebas

```bash
python -m unittest discover pruebas -v
```

Comprueban la geometría (ángulo del tronco, polígonos), que cada comportamiento del guion demo se detecte una sola vez
y en el momento correcto, que dos personas que se cruzan caminando no se marquen como pelea, que una persona tendida
desde el inicio no cuente como caída y que el gestor de alertas evite duplicados y guarde la evidencia.

## Estructura

```
vigia/
├── iniciar.bat, demo.bat, iniciar.sh   inicio con un clic
├── config.yaml                         cámaras, modelos, umbrales y privacidad
├── vigia/
│   ├── modelos.py      YOLO11 pose + seguimiento ByteTrack, objetos, rostros (YuNet) y expresión (MobileFaceNet)
│   ├── historial.py    memoria temporal de cada persona y objeto seguido
│   ├── reglas.py       violencia, caída, merodeo, intrusión, abandono, hurto y emociones, con su explicación
│   ├── camara.py       lectura de webcam/RTSP/archivo con reconexión y análisis en un hilo por cámara
│   ├── alertas.py      enfriamiento, base SQLite, capturas y envío al panel (Server-Sent Events)
│   ├── dibujo.py       anotaciones y pixelado de rostros
│   ├── simulador.py    escena sintética del modo demo
│   └── servidor.py     API y video MJPEG (Starlette)
├── web/                panel (HTML, CSS y JavaScript sin dependencias)
├── scripts/            descarga de modelos y evaluación con RWF-2000
├── pruebas/            pruebas automáticas
└── datos/              alertas.db, capturas y ajustes (se crea al ejecutar)
```

## Privacidad, ética y marco legal

- Instale cámaras con fines definidos y avisos visibles. En Perú rigen la Ley N.° 29733 de Protección de Datos
  Personales y la directiva de la Autoridad Nacional de Protección de Datos Personales sobre el tratamiento de datos
  mediante sistemas de videovigilancia.
- El análisis de **expresiones faciales** es el componente menos fiable: una expresión no equivale a una emoción y
  los modelos tienen sesgos según edad, etnia y condiciones de luz. La Ley de IA de la Unión Europea prohíbe inferir
  emociones en el trabajo y en la educación. Por eso aquí es informativo, se puede desactivar en **Ajustes** y su
  alerta viene apagada. Úselo solo con consentimiento.
- Las capturas y la base de datos quedan en la carpeta `datos/` de su equipo; nada se envía a internet.

## Limitaciones conocidas

- Las reglas son heurísticas sobre la pose: no aprenden de datos. Su precisión real debe medirse con el script de
  evaluación antes de afirmar cifras de desempeño.
- Abrazos, juegos bruscos o bailes cerca de otra persona pueden parecer peleas; agacharse a recoger algo puede
  parecer una caída si dura poco. Ajuste umbrales y tiempos según el lugar.
- El posible hurto depende de que el seguidor mantenga el ID del objeto; si el objeto se tapa por completo, se pierde.
- Con iluminación baja, oclusiones fuertes o personas muy lejanas (menos de unos 60 píxeles de alto) la pose
  se degrada, como advierte la literatura revisada.
- El modo demo usa personas sintéticas: prueba la lógica y el panel, no la precisión de los modelos.

## Solución de problemas

| Problema | Qué hacer |
|---|---|
| Va lento (menos de 5 fps) | En `config.yaml`, baje `tamano_imagen` a 320 y `fps_objetivo` a 6 |
| «No se pudo abrir 0» | Cierre otras apps que usen la webcam (Zoom, Teams) o pruebe con la fuente `1` |
| La cámara IP no conecta | Pruebe la URL en VLC (Medio → Abrir ubicación de red); verifique usuario, clave y que esté en la misma red |
| No aparecen emociones | Faltan los modelos ONNX: ejecute `python scripts/descargar_modelos.py` |
| Error al instalar `torch` | Use Python 3.10, 3.11 o 3.12 |

## Créditos

Modelos: YOLO11 de Ultralytics (licencia AGPL-3.0; para uso comercial requiere licencia de Ultralytics), YuNet y
el reconocedor de expresiones de OpenCV Zoo (Apache-2.0). Desarrollado con asistencia de Claude (Anthropic) como
producto de la revisión sistemática del autor.
