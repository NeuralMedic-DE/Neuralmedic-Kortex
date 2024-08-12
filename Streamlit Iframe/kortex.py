import streamlit as st
import configparser
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import numpy as np
import io
import time
import matplotlib.pyplot as plt

# Import VEP and SSVEP tools
from eeg_analyzer.vep.vep_tools import VEPAnalysisTools
from eeg_analyzer.vep.main import load_eeg_data as load_vep_data, check_equal_length as check_equal_length_vep, \
    find_increasing_indices, calculate_diff_list
from eeg_analyzer.ssvep.ssvep_tools import SSVEPAnalysisTools
from eeg_analyzer.ssvep.main import load_eeg_data as load_ssvep_data, preprocess_ssvep_data

# Hide Streamlit's default header and footer
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Load environment variables from .ini file
config = configparser.ConfigParser()
config.read('.ini')

# Retrieve API key from environment variable
google_api_key = config['api_key']['GOOGLE_API_KEY']

# Check if the API key is available
if google_api_key is None:
    st.warning("API key not found. Please set the GOOGLE_API_KEY in the .ini file.")
    st.stop()

# Use st.radio for mode selection
mode = st.radio("Select Analysis Mode", ["VEP", "SSVEP"])

# File Upload with unique keys
uploaded_file = st.file_uploader("Upload a Text File", type=["txt"], key=f"{mode}_file_uploader")

if uploaded_file is not None:
    st.text("Text File Uploaded Successfully!")

    # Read text file
    text_data = uploaded_file.read().decode("utf-8")

    if mode == "VEP":
        # Process VEP EEG data
        list_channels = load_vep_data(io.StringIO(text_data))

        # Check if all arrays in list_channels have the same length
        if not check_equal_length_vep(list_channels):
            st.error("Error: Not all arrays have the same length.")
        else:
            num_channels = len(list_channels)
            st.write(f"Number of channels: {num_channels}")

            # Find indices where the previous value is less than the current value
            indices = find_increasing_indices(list_channels[1])

            # Calculate the difference between consecutive indices
            diff_list = calculate_diff_list(indices)

            size_row = min(diff_list)  # You can adjust this based on your needs
            size_col = len(diff_list)
            st.write(f"Size Row: {size_row}, Size Col: {size_col}")

            # Extract a subset of the original array using the calculated indices
            eeg_data = list_channels[0][indices[0]:indices[0] + size_row]

            # Calculate number of columns for reshaping
            n_col = len(eeg_data) // size_row

            # Reshape the array
            reshaped_eeg_array = eeg_data[:int(n_col) * size_row]
            reshaped_eeg_array = reshaped_eeg_array.reshape(-1, size_row).transpose()

            reshaped_eeg_row_mean = np.mean(reshaped_eeg_array, axis=1)

            Fs = 5000  # Sampling frequency

            # Initialize the VEPAnalysisTools class
            vep_tools = VEPAnalysisTools(reshaped_eeg_row_mean, Fs)

            # Reshape data
            reshaped_data = vep_tools.reshape_data(size_row)

            # Button to display the VEP plot
            if st.button("Show VEP Plot"):
                st.write("Plotting VEP Data")
                try:
                    # Create a figure and axis
                    fig, ax = plt.subplots()
                    vep_tools.plot_vep(reshaped_data, ax=ax)
                    st.pyplot(fig)  # Pass the figure to st.pyplot
                except Exception as e:
                    st.error(f"Error plotting VEP data: {e}")

            # Define intervals for amplitude and latency calculation
            intervals = [
                {'name': 'P100', 't_min': 80, 't_max': 130, 'type': 'positive'},
                {'name': 'N75', 't_min': 55, 't_max': 85, 'type': 'negative'},
                {'name': 'N135', 't_min': 130, 't_max': 180, 'type': 'negative'}
            ]

            # Display amplitude and latency results
            st.write("Displaying Amplitude and Latency Results")
            analysis_results = vep_tools.display_amplitude_latency_results(reshaped_data, intervals)
            st.text(analysis_results)

    elif mode == "SSVEP":
        # Process SSVEP EEG data
        list_channels = load_ssvep_data(io.StringIO(text_data))

        # Check if all arrays in list_channels have the same length
        if not len(set(map(len, list_channels))) == 1:
            st.error("Error: Not all arrays have the same length.")
        else:
            num_channels = len(list_channels)
            st.write(f"Number of channels: {num_channels}")

            eeg_data = np.array(list_channels)
            Fs = 5000  # Sampling frequency
            preprocessed_data = preprocess_ssvep_data(eeg_data, Fs)

            reshaped_eeg_row_mean = np.mean(preprocessed_data, axis=1)

            # Initialize the SSVEPAnalysisTools class
            ssvep_tools = SSVEPAnalysisTools(reshaped_eeg_row_mean, Fs)

            # Reshape data
            size_row = preprocessed_data.shape[1]
            reshaped_data = ssvep_tools.reshape_data(size_row)

            # Button to display the SSVEP plot
            if st.button("Show SSVEP Plot"):
                st.write("Plotting SSVEP Data")
                try:
                    fig, ax = plt.subplots()
                    ssvep_tools.plot_ssvep(reshaped_data, ax=ax)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error plotting SSVEP data: {e}")

            # Define target frequencies for SSVEP analysis
            frequencies = [
                {'name': 'Frequency 1', 'frequency': 12},
                {'name': 'Frequency 2', 'frequency': 15}
            ]

            # Display PSD and SNR results
            st.write("Displaying PSD and SNR Results")
            analysis_results = ssvep_tools.display_psd_snr_results(reshaped_data, [f['frequency'] for f in frequencies])
            st.text(analysis_results)

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get User Question and display input box at the bottom
    with st.form(key='my_form', clear_on_submit=True):
        user_question = st.text_input("Ask a Question:", key="input_box")
        submit_button = st.form_submit_button("↵")

    if submit_button:
        if user_question:
            # Split the analysis results into chunks for embedding
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
            texts = text_splitter.split_text(analysis_results)

            # Calculate and display total amount of text
            total_text_length = sum(len(text) for text in texts)
            st.write(f"Total amount of text to be processed: {total_text_length} characters")


            # Function to process texts in smaller batches
            def process_texts_in_batches(texts, embeddings, batch_size=10, delay=1):
                all_embeddings = []
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    try:
                        batch_embeddings = embeddings.embed_documents(batch_texts)
                        all_embeddings.extend(batch_embeddings)
                    except Exception as e:
                        st.warning(f"Error embedding content: {e}. Retrying after a delay...")
                        time.sleep(delay)
                        batch_embeddings = embeddings.embed_documents(batch_texts)
                        all_embeddings.extend(batch_embeddings)
                return all_embeddings


            # Create embeddings using Google Generative AI Embeddings
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

            # Process texts in smaller batches to handle quota limits
            text_embeddings = process_texts_in_batches(texts, embeddings)

            # Create Chroma vector index from text embeddings
            vector_index = Chroma.from_texts(texts, embeddings).as_retriever()

            # Get Relevant Documents
            docs = vector_index.invoke(user_question)

            # Define Prompt Template
            prompt_template = """
            You are Gemini-Kortex who will aid in helping doctors understand EEG data from patients.
            Use your own knowledge to answer as well as context.
            Provide your thoughts in a table with the following columns:
            parameter, normal range, recorded, deduction

            Provide a concise overall condition of the patient from the values.

            \n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:
            """

            # Create Prompt
            context_response = " ".join([doc.page_content for doc in docs])
            prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

            # Load QA Chain
            model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7, google_api_key=google_api_key)
            chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

            # Combine context and analysis summary
            combined_context = context_response + "\n\n" + analysis_results

            # Get Response
            response = chain.invoke({"input_documents": docs, "question": user_question, "context": combined_context})

            # Store user question and assistant answer in chat history
            st.session_state.messages.append({"role": "user", "content": user_question})
            st.session_state.messages.append({"role": "assistant", "content": response['output_text']})

            # Display updated chat history
            st.rerun()

        else:
            st.warning("Please enter a question.")
