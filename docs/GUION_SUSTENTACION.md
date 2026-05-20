# 🎙️ Guión de Sustentación – RiskLab USTA
## Teoría del Riesgo · Python para APIs e IA
**Alejandra Sepúlveda · Ingrid Umbacia Ramírez | USTA 2025**

---

## 📋 Estructura de la Sustentación (20–25 min)

| Tiempo | Bloque | Responsable |
|--------|--------|-------------|
| 0–2 min | Introducción y contexto | Ambas |
| 2–8 min | Demo en vivo: tablero + /docs | Alejandra |
| 8–14 min | Explicación API: flujo, Pydantic, Depends, SQLAlchemy | Ingrid |
| 14–18 min | Teoría del Riesgo: interpretación de resultados | Ambas |
| 18–22 min | ML: Singleton, pipeline, /predict | Ingrid |
| 22–25 min | Preguntas del profesor | Ambas |

---

## 🎬 BLOQUE 1: INTRODUCCIÓN (0–2 min)

**Alejandra abre:**

> "Buenas [mañanas/tardes]. Somos Alejandra Sepúlveda e Ingrid Umbacia, y hoy les presentamos RiskLab USTA, nuestro sistema integral de análisis de riesgo financiero. El proyecto implementa los contenidos completos de los dos cursos: Teoría del Riesgo y Python para APIs e IA, articulados en cinco capas que van desde la ingesta de datos hasta el despliegue en producción con Docker y Render."

> "Analizamos un portafolio de seis activos: Motorola Solutions (MSI), ExxonMobil (XOM), Johnson & Johnson (JNJ), Procter & Gamble (PG), Unilever (UL) y TSMC (TSM), cubriendo tecnología, energía, salud, consumo básico y semiconductores."

---

## 🖥️ BLOQUE 2: DEMO EN VIVO (2–8 min)

### Script de demo (en este orden):

1. **Abrir el tablero Streamlit** en `localhost:8501`
   - Mostrar la navegación por tabs (no sidebar saturada)
   - Explicar: "La barra lateral solo tiene el selector de período. Los módulos están organizados en tres grupos de tabs para no saturar la interfaz."

2. **Tab "Módulos de Mercado" → Análisis Técnico**
   - Seleccionar TSM
   - Señalar el gráfico de 4 paneles: "precio con Bollinger, RSI, MACD y Estocástico"
   - Leer el semáforo: "Aquí vemos los indicadores del backend en tiempo real"

3. **Tab "Módulos de Mercado" → VaR/CVaR**
   - Dejar pesos equiponderados, conf = 95%
   - Clic en "Calcular VaR vía Backend"
   - Señalar los 3 métodos y el test de Kupiec

4. **Tab "API · ML · Macro" → API FastAPI**
   - Mostrar la tabla de endpoints
   - Ejecutar `/activos` en vivo

5. **Abrir pestaña del navegador: `localhost:8000/docs`**
   - "FastAPI genera automáticamente esta documentación Swagger UI"
   - Expandir `/var` → mostrar el Request Schema de Pydantic
   - Expandir `/opcion/precio` → mostrar campos S, K, T, r, sigma, tipo

6. **Abrir `localhost:8000/redoc`**
   - "Redoc es una vista alternativa más legible para documentación"

---

## ⚙️ BLOQUE 3: ARQUITECTURA API (8–14 min)

**Ingrid explica el flujo:**

> "El flujo de un request comienza cuando el tablero Streamlit llama a `requests.post()` con el cuerpo del request. FastAPI recibe la llamada en el endpoint `async def calcular_var(req: VaRRequest, ...)`. Lo primero que hace FastAPI es validar el request con Pydantic: verifica que los tickers existan, que los pesos sean no negativos y que sumen exactamente 1.0 con una tolerancia del 1%. Si alguna validación falla, FastAPI retorna automáticamente un HTTP 422 con detalle del error."

### Sobre Pydantic (demostrar en /docs):

> "Los modelos Pydantic con `@field_validator` y `@model_validator` hacen toda la validación de entrada. Por ejemplo, en `VaRRequest`, el `@field_validator('tickers')` normaliza los tickers a mayúsculas automáticamente. Y el `@model_validator(mode='after')` verifica que la longitud de tickers coincida con la de pesos, y que la suma sea 1.0."

### Sobre Depends():

> "Una vez validado el request, FastAPI inyecta las dependencias declaradas con `Depends()`. `DataService` se inyecta para obtener los precios, `RiskCalculator` para los cálculos de VaR, y `Session` de SQLAlchemy para la base de datos. La ventaja de `Depends()` es que los servicios se crean una sola vez y se reutilizan. Si pusiéramos la lógica directamente en la ruta, no habría separación de responsabilidades, no podríamos hacer tests unitarios y el código sería difícil de mantener."

### Sobre BaseSettings:

> "Las variables sensibles como `FRED_API_KEY` nunca se hardcodean en el código fuente. `Settings(BaseSettings)` las lee del archivo `.env`. El decorador `@lru_cache` en `get_settings()` garantiza que la instancia se crea una sola vez: es el mismo patrón Singleton que luego usamos para el modelo ML."

### Sobre SQLAlchemy ORM:

> "Para persistencia usamos SQLAlchemy ORM en lugar de SQL raw. Tenemos cuatro tablas: `PrecioCache` para cachear los precios de Yahoo Finance y evitar rate-limit, `MacroCache` para los indicadores macro, `PredictionLog` para auditar cada predicción del modelo ML con su versión, features, dirección y confianza, y `StressLog` para registrar los escenarios de stress testing. La ventaja del ORM sobre SQL raw es el tipado fuerte, mayor seguridad (no SQL injection) y la posibilidad de migraciones con Alembic."

---

## 📊 BLOQUE 4: TEORÍA DEL RIESGO (14–18 min)

### 4.1 Rendimientos y Hechos Estilizados

> "Los rendimientos logarítmicos se justifican por tres propiedades: aditividad temporal (log(P_t/P_0) = suma de log-rendimientos diarios), simetría (la subida del 10% y la bajada del 10% son simétricas en logs) y mejor aproximación a la normalidad para retornos pequeños. El test de Jarque-Bera rechaza la normalidad con p = [COMPLETAR], confirmando que los rendimientos tienen kurtosis [COMPLETAR] y colas más gruesas que la normal. Esto tiene implicación directa en el VaR: el VaR paramétrico normal **subestima** el riesgo real porque no captura la probabilidad de eventos extremos."

### 4.2 GARCH

> "Los modelos GARCH capturan el **agrupamiento de volatilidad**: períodos de alta volatilidad se agrupan. El GARCH(1,1) dice que la varianza de hoy depende de la varianza de ayer y del cuadrado del rendimiento de ayer. Según el criterio AIC y BIC, el mejor modelo para [TICKER] fue [MODELO], con AIC=[COMPLETAR]. Este modelo pronostica una volatilidad promedio de [COMPLETAR]% diario para los próximos 10 días, equivalente a [COMPLETAR]% anualizado."

> "EWMA con λ=0.94 es más simple: asigna mayor peso a observaciones recientes con decaimiento exponencial. La diferencia con GARCH es que GARCH tiene retorno a la media a largo plazo (parámetro omega), mientras que EWMA no converge a ningún nivel de largo plazo."

### 4.3 VaR y Kupiec

> "El VaR al 95% diario del portafolio equiponderado fue: paramétrico [COMPLETAR]%, histórico [COMPLETAR]% y Montecarlo [COMPLETAR]%. El VaR histórico es el más robusto dado el perfil leptocúrtico de los rendimientos, porque no asume ninguna distribución paramétrica."

> "El test de Kupiec evalúa si el número de violaciones (días donde la pérdida superó el VaR) es estadísticamente consistente con el nivel de confianza. Con confianza del 95%, esperamos que el 5% de los días el VaR sea violado. El estadístico LR sigue una chi-cuadrado con 1 grado de libertad. El valor crítico es 3.841. Nuestro LR=[COMPLETAR], p-valor=[COMPLETAR], por lo tanto [el modelo es válido / el modelo es rechazado]."

> "El CVaR (Expected Shortfall) es preferido por Basilea III porque es coherente: cumple la propiedad de subaditividad, es decir, el riesgo de un portafolio no puede superar la suma de los riesgos individuales."

### 4.4 CAPM y Markowitz

> "El activo con mayor riesgo sistemático es [TICKER] con beta=[COMPLETAR] y R²=[COMPLETAR], lo que significa que el [COMPLETAR]% de su varianza se explica por movimientos del S&P 500. El activo más defensivo es [TICKER] con beta=[COMPLETAR]."

> "En la frontera de Markowitz con 10,000 portafolios simulados, el portafolio de máximo Sharpe asigna [COMPLETAR]% a [TICKER] y reduce el peso de activos altamente correlacionados. La restricción de no negatividad (w≥0) elimina ventas en corto, haciendo la frontera más conservadora que la frontera sin restricciones."

### 4.5 Nelson-Siegel

> "El modelo Nelson-Siegel ajusta la curva de rendimientos con cuatro parámetros: β₀ es el nivel de largo plazo (la tasa a madurez infinita), β₁ determina la pendiente (diferencia entre corto y largo plazo), β₂ genera la joroba o curvatura, y λ controla qué tan rápido decae el factor de corto plazo. El ajuste se hace con scipy.optimize.least_squares con la restricción λ>0 para garantizar que el factor de decaimiento sea positivo."

### 4.6 Black-Scholes y Greeks

> "Black-Scholes asume volatilidad constante, mercado continuo, sin dividendos y tasa libre de riesgo constante. Valoramos opciones hipotéticas sobre los activos del portafolio usando el precio actual como subyacente S, la volatilidad estimada por GARCH como σ, y la Rf del endpoint /macro como r."

> "Las cinco Greeks: Delta mide sensibilidad al precio del subyacente. Para una call ATM, delta ≈ 0.5 (la opción tiene 50% de probabilidad de expirar in-the-money). Gamma mide la aceleración de Delta, es máxima cuando S≈K. Vega mide sensibilidad a la volatilidad implícita: es clave en stress testing porque un choque de volatilidad afecta directamente el precio de las opciones. Theta es el decaimiento temporal: siempre negativo para el comprador."

> "Verificamos la paridad put-call: C - P = S - K·e^(-rT). La diferencia obtenida fue [COMPLETAR] ≈ 0, confirmando la consistencia del modelo."

### 4.7 Stress Testing

> "Aplicamos tres escenarios de stress al portafolio óptimo: un shock de tasa +200 pb, una crisis de volatilidad estilo COVID (+30% en vol), y un crash de precio de -20%. Los resultados mostraron que el escenario más severo fue [COMPLETAR] con una pérdida estimada de [COMPLETAR]%, comparado con el VaR base de [COMPLETAR]%. Esto es [X] veces el VaR base, lo que refleja eventos de cola extrema no capturados por el VaR estándar."

---

## 🤖 BLOQUE 5: MACHINE LEARNING (18–22 min)

**Ingrid:**

> "El modelo ML predice la dirección del rendimiento a 5 días. Usamos RandomForestClassifier con 100 árboles en un pipeline con StandardScaler. Las features son: RSI normalizado (entre 0 y 1), MACD divido por precio, posición dentro de las Bandas de Bollinger (0=banda inferior, 1=banda superior), ratio SMA20/SMA50 (señal de cruce de medias), retorno acumulado 5 días, y volatilidad histórica 20 días."

> "Para evitar data leakage temporal usamos train_test_split con shuffle=False. Esto garantiza que el modelo nunca ve datos futuros durante el entrenamiento: entrenamos con el 70% más antiguo y evaluamos con el 30% más reciente."

> "El patrón Singleton garantiza que el modelo se carga una sola vez en memoria. Pueden ver en los logs que el mensaje 'Modelo ML cargado' aparece solo una vez, aunque hagamos múltiples llamadas a /predict. El `__new__()` de Python controla la creación de instancias: si `_instance` ya existe, retorna esa misma instancia sin crear una nueva. Esto es crítico en APIs de producción para no recaugar el modelo en cada request."

---

## ❓ PREGUNTAS FRECUENTES Y RESPUESTAS

### 🔵 Python / Arquitectura

**P: ¿Por qué eligieron esos activos?**
> "Elegimos activos de sectores diversos para tener correlaciones bajas: tecnología (MSI, TSM), energía (XOM), salud (JNJ), consumo básico (PG, UL). Esto maximiza los beneficios de diversificación en la frontera de Markowitz."

**P: ¿Cómo validaron los datos de entrada con Pydantic? Muestre un @field_validator.**
> "En VaRRequest tenemos `@field_validator('tickers')` que normaliza a mayúsculas, y `@model_validator(mode='after')` que verifica que len(tickers)==len(weights) y que abs(sum(weights)-1.0)<=0.01. Si falla, FastAPI retorna 422 automáticamente con el detalle del error."

**P: ¿Qué dependencias inyectan con Depends()? ¿Por qué no pusieron la lógica en la ruta?**
> "Inyectamos DataService, RiskCalculator, PortfolioAnalyzer, OptionPricer, StressTester, MLModelSingleton y Session de BD. Separamos la lógica en servicios porque: 1) facilita los tests unitarios (podemos mockear el DataService), 2) evita repetición de código, 3) permite que el @lru_cache cree los servicios solo una vez."

**P: ¿Dónde están las API keys? ¿Qué pasa si la API externa no responde?**
> "Las API keys están en el archivo .env que está en .gitignore. Settings(BaseSettings) las lee del .env. Si Yahoo Finance no responde, el DataService lanza un RuntimeError que main.py convierte en HTTPException 503 con detalle descriptivo."

**P: ¿Cómo están persistiendo los datos? ¿Por qué SQLAlchemy ORM en lugar de SQL Raw?**
> "SQLite con SQLAlchemy ORM. Elegimos ORM porque: tipado fuerte con Python, queries como objetos (más legibles), mayor seguridad (sin SQL injection), y la sesión se inyecta con Depends(get_db)."

**P: ¿Cómo verifican que el Singleton no recarga el modelo?**
> "En los logs del backend al hacer dos llamadas a /predict, el mensaje 'Modelo ML cargado' aparece solo una vez. El `_instance` es un atributo de clase: si ya está asignado, `__new__` retorna esa misma instancia."

**P: ¿Por qué multi-stage en el Dockerfile?**
> "El Stage 1 instala build-essential, gcc y compila todos los paquetes con pip. El Stage 2 copia solo los paquetes compilados de /install. Esto reduce la imagen de ~600 MB a ~150 MB porque no incluye los compiladores ni las dependencias de build en la imagen final. Un cold start de Render con 600 MB sería inaceptable para la demo en vivo."

**P: ¿Qué pasa si los tests fallan en el CI?**
> "GitHub Actions para el workflow: el step de Docker build no se ejecuta si pytest falla, y el deploy en Render no recibe el push. Esto garantiza que solo código validado llega a producción."

---

### 🔴 Teoría del Riesgo

**P: ¿Por qué eligieron log-rendimientos en lugar de rendimientos simples?**
> "Los log-rendimientos tienen tres ventajas: 1) Aditividad temporal: el rendimiento acumulado es la suma de los diarios. 2) Simetría: +10% y -10% son simétricamente opuestos. 3) Aproximación a la normalidad para retornos pequeños, necesaria para el VaR paramétrico."

**P: ¿Qué modelo GARCH seleccionaron y por qué? ¿Cómo cambia entre EWMA y GARCH?**
> "Seleccionamos [MODELO] porque tuvo el menor AIC/BIC. EWMA es más simple pero no tiene retorno a la media a largo plazo: la volatilidad EWMA puede alejarse indefinidamente. GARCH tiene un parámetro omega que ancla la varianza a un nivel de largo plazo, haciendo las predicciones más estables para horizontes largos."

**P: ¿Qué significa el backtesting de Kupiec? ¿Cuál de sus tres VaR pasa el test al 95%?**
> "Kupiec prueba si la tasa de violaciones observada es estadísticamente igual a la tasa esperada. Si el VaR al 95% es correcto, esperamos que el 5% de los días sea violado. El estadístico LR=-2*(log-verosimilitud bajo H0 - log-verosimilitud bajo HA) sigue chi²(1). Nuestro [MÉTODO] pasó el test con LR=[COMPLETAR] y p-valor=[COMPLETAR] > 0.05."

**P: Para Markowitz, ¿cómo cambia el portafolio óptimo con no negatividad? ¿Qué activos quedan con peso cero?**
> "Con la restricción w≥0 no puede haber ventas en corto. La frontera eficiente se hace más conservadora: el portafolio óptimo puede excluir activos con alta correlación o bajo Sharpe individual. En nuestro caso, [COMPLETAR] quedó con peso ≈0 porque su retorno esperado es bajo en relación a su volatilidad."

**P: ¿Qué significa la duración? ¿Cuánto se mueve el precio del bono ante un shock de +200 pb?**
> "La duración de Macaulay es el promedio ponderado del tiempo a recibir los flujos, medida en años. La duración modificada = D_Mac/(1+y/m) mide la sensibilidad porcentual: ΔP/P ≈ -D_mod × Δy. Para nuestro bono de ejemplo con D_mod=[COMPLETAR] años, un shock de +200 pb produce ΔP/P ≈ -[COMPLETAR]×0.02 = [COMPLETAR]%. La convexidad corrige este estimado lineal: el precio real con +200 pb fue $[COMPLETAR]."

**P: Para opciones, verifiquen numéricamente la paridad put-call. ¿Qué es vega y para qué sirve en stress testing?**
> "La paridad put-call dice C - P = S - K·e^(-rT). Con nuestros parámetros: Call=[COMPLETAR], Put=[COMPLETAR], S-K·e^(-rT)=[COMPLETAR]. La diferencia es [COMPLETAR]≈0. Vega mide el cambio de precio de la opción por un aumento del 1% en la volatilidad implícita. En stress testing, cuando aplicamos el escenario de volatilidad +30%, el precio de las calls sube porque la volatilidad implícita sube. Vega cuantifica ese impacto."

**P: ¿Qué señales está generando el sistema hoy?**
> "[COMPLETAR con resultados reales del endpoint /alertas] Según el backend, los activos con señal de COMPRA son [COMPLETAR] y los de VENTA son [COMPLETAR]. La señal se determina por mayoría de votos entre 5 indicadores: RSI, MACD, Bollinger, SMA y Estocástico."

**P: ¿Cómo se compara la pérdida bajo el peor escenario de stress vs el VaR base?**
> "El peor escenario fue [COMPLETAR] con una pérdida de [COMPLETAR]%, comparado con el VaR histórico base de [COMPLETAR]%. Esto representa [COMPLETAR]x el VaR base. Esta diferencia refleja que el VaR no captura eventos de cola extrema (por eso Basilea III requiere el stressed VaR para los bancos)."

---

### 🟡 Política de IA – Preguntas Sugeridas

**P: ¿Usaron inteligencia artificial en el desarrollo? ¿Cómo?**
> "Sí, utilizamos Claude de Anthropic como asistente de desarrollo. Lo usamos para: estructurar el código inicial del backend (las clases de servicios), generar las fórmulas matemáticas de Black-Scholes y Nelson-Siegel en código Python, y diseñar los tests unitarios con TestClient. Sin embargo, todo el código fue revisado, entendido y adaptado por nosotras. Las decisiones metodológicas (selección de modelos GARCH, parámetros CAPM, diseño del pipeline ML, selección del portafolio) son completamente nuestras."

**P: ¿Pueden explicar cualquier línea del código que generó la IA?**
> "Sí. [Señalar cualquier parte del código y explicar línea por línea. Por ejemplo: el `@field_validator` en models/schemas.py, el cálculo de d1 y d2 en services/derivatives.py, o el patrón `__new__` del Singleton en ml/model.py]"

**P: ¿Qué harían diferente si no hubieran usado IA?**
> "El tiempo de desarrollo habría sido significativamente mayor, especialmente en las partes más matemáticas como Nelson-Siegel y Black-Scholes. Pero el aprendizaje conceptual fue igual: tuvimos que estudiar cada modelo para poder interpretar los resultados, defender las decisiones metodológicas y diseñar los tests. La IA fue un acelerador, no un sustituto del aprendizaje."

**P: ¿Documentaron el uso de IA en el README?**
> "Sí, en el README hay una sección 'Uso de Herramientas de IA' que describe específicamente qué partes se desarrollaron con apoyo de IA y qué partes son de nuestra autoría, como lo establece la política del curso."

---

## 📌 SCRIPT DE DEMO (ORDEN RECOMENDADO)

```
1. localhost:8501 → Tab "Módulos de Mercado" → Análisis Técnico (TSM)
2. localhost:8501 → Tab "Módulos de Mercado" → VaR/CVaR → Calcular
3. localhost:8501 → Tab "Módulos de Mercado" → Señales
4. localhost:8501 → Tab "Renta Fija" → Opciones B-S (MSI, call ATM)
5. localhost:8501 → Tab "Renta Fija" → Stress Testing → Ejecutar
6. localhost:8501 → Tab "API · ML" → API FastAPI → Ver tabla + ejecutar /activos
7. localhost:8000/docs → Expandir /var → Mostrar schema
8. localhost:8000/docs → Expandir /predict → Ejecutar en vivo
9. localhost:8501 → Tab "API · ML" → ML → Predecir (MSI, 5 días)
10. localhost:8000/redoc (mencionar brevemente)
```

---

## ⚠️ CHECKLIST ANTES DE LA SUSTENTACIÓN

- [ ] Backend corriendo en `localhost:8000` (verificar `/docs`)
- [ ] Frontend corriendo en `localhost:8501`
- [ ] Deploy en Render activo (hacer `curl <url>/` como calentamiento)
- [ ] Tener los valores reales del backend memorizados: VaR%, beta por activo, modelo GARCH ganador
- [ ] Verificar que los valores del semáforo de señales son legibles (texto oscuro sobre fondos de color)
- [ ] Abrir pestaña de `/docs` y `/redoc` en el navegador
- [ ] Practicar el script de demo sin quedarse sin tiempo

---

*Alejandra Sepúlveda · Ingrid Umbacia Ramírez | USTA 2025*
