import streamlit as st
import cancer_data_script
import additional_funcs
import pandas as pd
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class App:
    def __init__(self):
        self.script = cancer_data_script.Script()

    def run(self):
        self.Init_Streamlit_Page()

    def Init_Streamlit_Page(self):
        st.set_page_config(page_title="Breast Cancer Script")
        st.title('YZUP Python Project')
        st.markdown("---")

        data_load_state = st.text("You haven't uploaded a file yet")

        self.script.classifier_name = st.sidebar.selectbox(
            'Select Classifier',
            ('KNN', 'SVM', 'Naive Bayes')
        )
        #
        self.script.uploaded_file = st.sidebar.file_uploader('Choose a File!', type=['csv', 'xlsx'])
        if self.script.uploaded_file is not None:
            data_load_state.text("Loading data...")
            file_name = self.script.uploaded_file.name
            if file_name.endswith('.csv'):
                self.script.dataframe = pd.read_csv(self.script.uploaded_file)
            elif file_name.endswith('.xlsx'):
                self.script.dataframe = pd.read_excel(self.script.uploaded_file)

            st.write(self.script.dataframe[:10])
            data_load_state.text(f"Loading data...Done! {self.script.uploaded_file.name}")

            # Veri ön işlemeye gönderiliyor
            self.script.data_preprocess(self.script.dataframe)
            st.write("Shape of dataset:", self.script.X.shape)
            st.write("Number of classes:", len(np.unique(self.script.y)))
            st.write(f"Classes= M (Malignant): {np.sum(self.script.y == 1)}, B (Benign): {np.sum(self.script.y == 0)}")

            # korelasyon matrisi ve scatter grafigi cizdiriliyor
            st.write("\n")
            st.title("Correlation Matrix")
            st.markdown("---")
            corr_matrix = additional_funcs.plot_correlation(self.script.preprocessed_dataframe)
            # plt.show()
            st.pyplot(corr_matrix.get_figure())

            # Scatter Plot
            st.write("\n")
            st.title("Scatter Plot")
            st.markdown("---")
            scatter_plot = additional_funcs.plot_scatter(self.script.preprocessed_dataframe)
            # plt.show()
            st.pyplot(scatter_plot)

            st.write("\n")
            st.title("Training")
            st.markdown("---")
            learning_state = st.text("Training is carried out...")
            learning_time = st.text("")
            start_time = time.time()
            model = None
            if self.script.classifier_name == 'Naive Bayes':
                model = self.script.learn_with_bayes()
            else:
                model = self.script.gridsearch()
            self.script.evaluate_model(model)
            self.evaluate()
            end_time = time.time()
            hours, minutes, seconds = additional_funcs.format_time(end_time - start_time)
            learning_state.text("Training is carried out...Done!")
            learning_time.text(f"Elapsed time: {hours}:{minutes}:{round(seconds, 2)}")

    def evaluate(self):
        st.text(f"Classifier: {self.script.classifier_name}")
        st.write(f"""
                <table>
                    <tr>
                        <th>accuracy_score</th>
                        <th>precision_score</th>
                        <th>recall_score</th>
                        <th>f1_score</th>
                    </tr>
                    <tr>
                        <td>{accuracy_score(self.script.y_test, self.script.y_preds)}</td>
                        <td>{precision_score(self.script.y_test, self.script.y_preds)}</td>
                        <td>{recall_score(self.script.y_test, self.script.y_preds)}</td>
                        <td>{f1_score(self.script.y_test, self.script.y_preds)}</td>
                    </tr>
                </table>
                """, unsafe_allow_html=True)

        st.write("\n\n")

        fig = plt.figure(figsize=(8, 6))
        sns.heatmap(self.script.cf_matrix, annot=True)
        plt.xlabel("y_pred")
        plt.ylabel("y_true")
        st.pyplot(fig)
