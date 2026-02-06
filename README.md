# Silent Signals 🔍⌨️🖱️
### AI System to Detect Early Burnout & Stress Using Everyday Keyboard & Mouse Behavior

---

## 📌 Overview
Silent Signals is an experimental, privacy-first system that explores the use of **keyboard and mouse interaction patterns** to identify **early behavioral changes associated with burnout and chronic stress**.

Unlike traditional approaches that rely on surveys, wearables, or invasive monitoring, this system operates **passively and locally**, focusing on **behavioral trend detection rather than diagnosis or labeling**.

---

## 💡 What It Does
- Collects **typing speed, pause patterns, error rate, and mouse movement**
- Extracts behavioral features using **sliding time windows**
- Models a **personal baseline** for each user
- Detects **deviations from normal interaction patterns**
- Flags **early burnout risk trends**, not medical labels

---

## 🧠 Why This Matters
Burnout often develops gradually and goes unnoticed until productivity and well-being are significantly affected.  
Silent Signals aims to demonstrate how **subtle behavioral changes** in everyday interactions can provide **early awareness**, while maintaining strong privacy and ethical boundaries.

---

## 🛠️ Tech Stack

| Component | Tool |
|--------|------|
| Data Collection | Python (`pynput`) |
| Feature Extraction | NumPy, Pandas |
| Machine Learning | Scikit-learn (KNN / Isolation Forest) |
| Visualization | Matplotlib |
| Interface | Command Line (CLI) |

---

## ⚖️ Ethical Considerations
- This system **does not diagnose mental health conditions**
- No content, keystrokes, or personal data are stored
- Designed strictly as a **research and self-awareness tool**
- Not intended for employee monitoring or surveillance

---

## 🚧 Project Status
This project is an **exploratory prototype** focused on feasibility and experimentation.  
It serves as a foundation for further research in **privacy-preserving behavioral analytics**.

---

## 📜 License
This project is licensed under the **MIT License**.
