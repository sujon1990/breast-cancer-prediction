import pickle
import streamlit as st

model = pickle.load(open('model/best_model', 'rb'))
# print(model.predict([[5,1,1,1,2,1,3,1,1]]))

def predict_breast_cancer(x):
    prediction = model.predict([x])
    return prediction

def main():
    st.title('Breast Cancer Prediction APP')
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Predict Breast Cancer or not </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    clump_thickness = st.text_input("clump_thickness","")
    size_uniformity = st.text_input("size_uniformity","")
    shape_uniformity = st.text_input("shape_uniformity","")
    marginal_adhesion = st.text_input("marginal_adhesion","")
    epithelial_size = st.text_input("epithelial_size","")
    bare_nucleoli = st.text_input("bare_nucleoli","")
    bland_chromatin = st.text_input("bland_chromatin","")
    normal_nucleoli = st.text_input("normal_nucleoli","")
    mitoses = st.text_input("mitoses","")
    x = (clump_thickness,size_uniformity,shape_uniformity,
                                       marginal_adhesion,
                                       epithelial_size,
                                       bare_nucleoli,
                                       bland_chromatin,
                                       normal_nucleoli,
                                       mitoses)
    result=""

    if st.button("Predict"):
        try:
            result = predict_breast_cancer(x)
            if result[0] == 2:
                st.success('Benign')
            else:
                st.warning('Malignant')

        except ValueError:
            st.success('Invalid Input. Enter Number to each input!')

    if st.button("About"):
        st.text("My APP")
        st.text("Sujon Ahmed")

if __name__=='__main__':
    main()