import streamlit as st
from detector import detect

st.title("AI Image Detector")

file = st.file_uploader("Upload Image")

if file:
    with open("input.jpg", "wb") as f:
        f.write(file.read())

    st.image("input.jpg")

    if st.button("Analyze"):
        scores, final, result , exif_data = detect("input.jpg")

        st.subheader("Scores")
        st.write(scores)
        st.write("EXIF DETAILS:", exif_data)
        st.subheader("Result")
        st.write(result)

        st.subheader("AI Probability")
        st.write(round(final * 100, 2), "%")
        st.write(final)
        