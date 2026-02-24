import pandas as pd
import pickle
import gradio as gr

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

def predict(gender, age, estimated_salary):
    gender = 1 if gender == "Male" else 0
    inu = pd.DataFrame([{
        "Gender": gender,
        "Age": age,
        "EstimatedSalary": estimated_salary
    }])

    result = model.predict(inu)[0]

    return "Purchased" if result == 1 else "Not Purchased"

app = gr.Interface(
    fn=predict,
    inputs=[
        gr.Radio(["Male", "Female"], label="Gender"),
        gr.Number(label="Age"),
        gr.Number(label="Estimated Salary")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Social Network Ads - Purchase Prediction",
    description="Prediction purchased a particular product"
)

if __name__ == "__main__":
    app.launch()
