import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from PIL import Image


# --------------------------------------
# Create a session for your details in the sidebar
st.sidebar.title("Session Details")
if 'name' not in st.session_state:
    st.session_state['name'] = "Ashfaq Khan"
if 'role' not in st.session_state:
    st.session_state['role'] = "Data Science Intern"
if 'batch' not in st.session_state:
    st.session_state['batch'] = "Elite - 21"    
if 'company' not in st.session_state:
    st.session_state['company'] = "Innomatics Research Labs"

# Display your details in the sidebar
st.sidebar.write(f"**Name:** {st.session_state['name']}")
st.sidebar.write(f"**Role:** {st.session_state['role']}")
st.sidebar.write(f"**Batch:** {st.session_state['batch']}")
st.sidebar.write(f"**Company:** {st.session_state['company']}")

# --------------------------------------



# Load the pickle files
with open('Balanced_RandomForest_multiclass.pkl', 'rb') as f:
    model_bagging_multiclass = pickle.load(f)

# Load and display logo image
image = Image.open("innomatics_research_labs_logo.png")
st.image(image, caption="", use_column_width=True)

# Set up Streamlit app title and sidebar
st.title('Machine Failure Prediction')

st.image("machine-animation.gif", caption="Loading GIF...", use_column_width=True)
st.image("Assemblage_Imprimante_3D.gif", caption="Loading GIF...", use_column_width=True)



st.sidebar.header('Input Features')

# Create input fields for all features used during training
air_temperature = st.sidebar.number_input('Air_temperature', value=298.10)
process_temperature = st.sidebar.number_input('Process_temperature', value=308.60)
rotational_speed = st.sidebar.number_input('Rotational_speed', value=1551.00)
tool_wear = st.sidebar.number_input('Tool_wear', value=3.00)
torque = st.sidebar.number_input('Torque', value=42.80)

# Include the categorical feature
type_input = st.sidebar.selectbox('Type_encoded', options=['M', 'L', 'H'])  # Example options based on your data

# Create a DataFrame for the input features
input_data = pd.DataFrame({
    'Air_temperature': [air_temperature],
    'Process_temperature': [process_temperature],
    'Rotational_speed': [rotational_speed],
    'Torque': [torque],
    'Tool_wear': [tool_wear],
    'Type_encoded': [type_input],  # Categorical feature
})

# Convert categorical feature to numerical (if needed)
le = LabelEncoder()
input_data['Type_encoded'] = le.fit_transform(input_data['Type_encoded'])

if st.button('Predict'):
    try:
        # Predict using model_bagging_multiclass
        prediction = model_bagging_multiclass.predict(input_data)
        st.write(f'Predicted Failure Type: {prediction[0]}')

        # Image and description mapping for each failure type
        image_map = {
            0.0: {'image': 'good machine.jpg', 'description': 'No failure detected. Machine is working optimally.', 'maintenance': 'No immediate maintenance required, continue regular checkups.'},
            1.0: {'image': 'real_heat_dissipation.jpg', 'description': 'Heat dissipation failure detected.', 'maintenance': 'Check cooling systems and ensure proper ventilation. Inspect and replace any worn-out heat sinks.'},
            2.0: {'image': 'power_failure_2_0.jpg', 'description': 'Power failure detected.', 'maintenance': 'Inspect electrical connections and power supply. Replace or repair faulty wiring.'},
            3.0: {'image': 'over_strain.jpg', 'description': 'Overstrain failure detected.', 'maintenance': 'Reduce load on the machine and check for any component overload. Ensure proper lubrication of parts.'},
            4.0: {'image': 'tool_wear_ok.jpg', 'description': 'Tool wear failure detected.', 'maintenance': 'Replace worn-out tools and calibrate the machine for accurate operations.'}
        }

        # Display the image, description, and maintenance details
        if prediction[0] in image_map:
            img = Image.open(image_map[prediction[0]]['image'])
            st.image(img, caption=f'Failure Type: {prediction[0]}', use_column_width=True)
            st.write(f"**Description:** {image_map[prediction[0]]['description']}")
            st.write(f"**Maintenance Suggestion:** {image_map[prediction[0]]['maintenance']}")
        else:
            st.write("No image available for this failure type.")


         # Add trainer and mentor images with names
        st.write("### Trainers and Mentors")
        trainer_image = Image.open("nagarajusir.jpg")
        mentor_image = Image.open("maam.jpg")
        st.image(trainer_image, caption="Trainer: Nagaraju Ekkirala", use_column_width=True)
        st.image(mentor_image, caption="Mentor: Lakshmi Teja Illuri", use_column_width=True)

        # Thank you salutation
        st.write("### Thank You for using the Machine Failure Prediction App!")   

    except Exception as e:
        st.error(f'Error during prediction: {e}')
