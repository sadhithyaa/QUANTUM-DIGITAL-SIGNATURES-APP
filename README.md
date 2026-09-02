# Egreen Quanta

A working local Streamlit prototype for quantum-inspired cyber-threat detection around digital signatures. It signs and verifies a user-provided message with ECDSA P-256, measures signature entropy, verification latency, failures, payload size, and timing variance, then runs the measurements through a hybrid graph-neural and quantum-inspired classifier.

## Run locally

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

The neural network is trained when the app first starts. Its default data is a generated bootstrap dataset so that the training path can be demonstrated immediately. Replace this with labelled telemetry from the target signing environment before using a classifier result for security decisions.

## Deploy with Streamlit Community Cloud

1. Add `app.py`, `requirements.txt`, and this `README.md` to the root of your GitHub repository.
2. Commit and push the files to GitHub.
3. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
4. Select the repository, choose its branch, set the main file path to `app.py`, then select **Deploy**.

Streamlit Community Cloud runs Python on its hosted server. Do not enter private signing keys or confidential transaction payloads into a public deployment.

## Important boundary

The cryptographic signing, verification, malformed-signature check, and runtime measurements are real operations on the machine running Streamlit. The app cannot inspect ECDSA's internal nonce or automatically detect remote side channels; those require instrumentation and labelled data from the system being monitored.
