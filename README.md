# PNCare-AI API

REST API for the PNCare-AI model (risk of no postnatal check by a medically trained provider within 48 hours
of birth, Bangladesh; soft-voting model trained on BDHS 2017-18 and 2022). It uses the same model file and the
same logic as the PNCare-AI Streamlit app.

| Method | Path | Returns |
|---|---|---|
| GET | `/health` | model status |
| GET | `/api/inputs` | the 12 inputs and allowed values |
| POST | `/api/predict` | risk, risk level (threshold 0.27), WHO-referenced actions, SHAP reasons, 22 selected features |
| POST | `/api/predict/batch` | risk and risk level for up to 20,000 mothers (JSON) |
| POST | `/api/upload` | same, from a CSV file with the 12 input columns |
| POST | `/api/preprocess` | checked inputs and the 22 selected features the model uses |
| GET | `/api/model/{version}` | model information (`latest`) |

Interactive documentation: `/docs`. Decision support only; inputs are processed in memory and are not stored.
