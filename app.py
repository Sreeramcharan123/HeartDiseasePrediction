from flask import Flask, render_template, request, send_file
import numpy as np
import pickle

# PDF imports
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER

app = Flask(__name__)

# Load model
model = pickle.load(open("heart_model.pkl", "rb"))

# Store last report
last_report = {}

# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():
    global last_report

    try:
        data = request.form.to_dict()

        name = data["name"]
        age = data["age"]
        sex = data["sex"]

        features = [
            float(data["age"]),
            float(data["sex"]),
            float(data["cp"]),
            float(data["trestbps"]),
            float(data["chol"]),
            float(data["thalach"])
        ]

        final = np.array([features])

        prediction = model.predict(final)[0]
        probability = model.predict_proba(final)[0][1] * 100
        risk = round(probability, 2)

        # Status + Advice
        if prediction == 1:
            status = "High Risk"
            advice_list = [
                "Exercise at least 30 minutes daily",
                "Avoid junk and oily food",
                "Reduce stress using meditation",
                "Avoid smoking and alcohol",
                "Consult a doctor regularly"
            ]
        else:
            status = "Low Risk"
            advice_list = [
                "Maintain a healthy balanced diet",
                "Stay physically active",
                "Sleep 7-8 hours daily",
                "Stay hydrated",
                "Continue regular checkups"
            ]

        # HTML advice
        advice_html = "<br>".join(advice_list)

        # Save for PDF
        last_report = {
            "name": name,
            "age": age,
            "gender": "Male" if sex == "1" else "Female",
            "status": status,
            "risk": risk,
            "advice_list": advice_list
        }

        return render_template(
            "index.html",
            prediction_text=status,
            risk=risk,
            name=name,
            advice=advice_html
        )

    except Exception as e:
        print(e)
        return render_template("index.html", prediction_text="❌ Invalid Input")


# ---------------- DOWNLOAD PDF ----------------
@app.route("/download")
def download():

    doc = SimpleDocTemplate("Heart_Report.pdf")
    styles = getSampleStyleSheet()

    # Title style center
    title_style = styles['Title']
    title_style.alignment = TA_CENTER

    content = []

    # Title
    content.append(Paragraph("🩺 Heart Health Report", title_style))
    content.append(Spacer(1, 20))

    # Table Data
    patient_data = [
        ["Name", last_report.get("name")],
        ["Age", last_report.get("age")],
        ["Gender", last_report.get("gender")],
        ["Result", last_report.get("status")],
        ["Risk Level", f"{last_report.get('risk')} %"]
    ]

    table = Table(patient_data, colWidths=[130, 220])

    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))

    content.append(table)
    content.append(Spacer(1, 25))

    content.append(Paragraph("📊 Risk Analysis", styles['Heading2']))
    content.append(Spacer(1, 10))

    risk = last_report.get("risk")

    if risk < 30:
        interpretation = "Low Risk – Maintain healthy lifestyle"
    elif risk < 70:
        interpretation = "Moderate Risk – Take precautions"
    else:
        interpretation = "High Risk – Consult a doctor"

    content.append(Paragraph(f"Risk Percentage: {risk}%", styles['Normal']))
    content.append(Paragraph(f"Interpretation: {interpretation}", styles['Normal']))

    content.append(Spacer(1, 20))

    # Recommendations
    content.append(Paragraph("💡 Recommendations", styles['Heading2']))
    content.append(Spacer(1, 10))

    for item in last_report.get("advice_list", []):
        content.append(Paragraph(f"• {item}", styles['Normal']))

    content.append(Spacer(1, 20))

    # Footer
    content.append(Paragraph(
        "Generated using AI-based Heart Disease Prediction System.",
        styles['Italic']
    ))

    doc.build(content)

    return send_file("Heart_Report.pdf", as_attachment=True)


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)