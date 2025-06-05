import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px

# Set page configuration, including a wide layout
st.set_page_config(
    page_title="Politische Text-Erkennung",
    layout="wide", # You can also use "wide" if you prefer a wider layout
    initial_sidebar_state="auto"
)

# Custom CSS for background color
# We use st.markdown with unsafe_allow_html=True to inject CSS
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f0f0; /* Page background color */
    }

    /* Title styling */
    h1 {
        font-family: 'Georgia', serif;
        color: #333333;
        font-size: 3em;
    }

    /* Style for the text input box itself */
    .stTextArea > div > div > textarea {
        background-color: #ffffff;
        border: 1px solid #cccccc;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize session state variables
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "current_chart_display" not in st.session_state:
    st.session_state.current_chart_display = "chart1"

data_class_avg = {
    "Junge Welt": [0.1, 0.2, 0.3, 0.15, 0.25],    
    "Nd": [0.2, 0.1, 0.25, 0.15, 0.3],            
    "Jacobin": [0.15, 0.25, 0.1, 0.2, 0.3],       
    "Tagesschau": [0.22, 0.18, 0.2, 0.2, 0.2],    
    "Taz": [0.17, 0.23, 0.3, 0.1, 0.2],           
    "Junge Freiheit": [0.05, 0.15, 0.3, 0.2, 0.3],
    "Tichys Einblick": [0.1, 0.1, 0.1, 0.3, 0.4]  
}

data_article_avg = {
    "Junge Welt": [0.2, 0.2, 0.2, 0.2, 0.2],
    "Nd": [0.2, 0.2, 0.2, 0.2, 0.2],
    "Jacobin": [0.2, 0.2, 0.2, 0.2, 0.2],
    "Tagesschau": [0.2, 0.2, 0.2, 0.2, 0.2],
    "Taz": [0.2, 0.2, 0.2, 0.2, 0.2],
    "Junge Freiheit": [0.2, 0.2, 0.2, 0.2, 0.2],
    "Tichys Einblick": [0.2, 0.2, 0.2, 0.2, 0.2]
}

classes = ["left", "left-center", "center", "right-center", "right"]

def dict_to_pdf(data_dict):
    dataframe_rows = []
    for medium, values_array in data_dict.items():
        for i, value in enumerate(values_array):
            row = {
                "Medium": medium,
                "Class": classes[i],
                "Value": value*100
            }
            dataframe_rows.append(row)
    data_df = pd.DataFrame(dataframe_rows)
    data_df["Percentage"] = data_df["Value"].apply(lambda x: f"{x:.1f}%")

    return data_df

df_media = dict_to_pdf(data_class_avg)
df_article = dict_to_pdf(data_article_avg)

colors_classes = [
    "rgba(54, 162, 235, 0.8)",
    "rgba(120, 175, 213, 0.8)",
    "rgba(185, 187, 190, 0.8)",
    "rgba(220, 143, 161, 0.8)",
    "rgba(255, 99, 132, 0.8)"
]

def create_stacked_chart(*, df, chart_title):
    """Creates the political leaning bar chart."""
    fig = px.bar(
        df,
        title=chart_title,
        x="Value",
        y="Medium",
        color="Class",
        orientation="h",
        barmode="stack",
        text="Percentage",
        color_discrete_sequence=colors_classes
    )

    fig.update_layout(
        xaxis_visible=False,
        xaxis_showticklabels=False,
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False,
        yaxis_title=None,
        paper_bgcolor='#f0f0f0',
        plot_bgcolor='#f0f0f0',
        yaxis=dict(
            tickfont=dict(
                size=18,
                weight='bold'
            )
        ),
    )

    fig.update_traces(
        textposition='inside',
        textfont=dict(color='white', size=15, weight='bold'),
        insidetextanchor='middle',
        hovertemplate="<b>Medium:</b> %{y}<br>" +
              "<b>Class:</b> %{data.name}<br>" +
              "<b>Percentage:</b> %{text}" +
              "<extra></extra>"
    )
    return fig

col_left_spacer, col_content, col_right_spacer = st.columns([1, 10, 1])

with col_content:
    # Title of the application
    st.title("Text Erkennung - Politische Neigung")

    # whitespace
    st.text("")

    user_input = st.text_area(
        "Gib einen Text ein:",
        height=200,
        placeholder="Dein politischer Text...",
        key="user_input"
    )

    # Create two columns for the buttons
    col1, col2 = st.columns(2)

    with col1:
        analyze_clicked = st.button("Analysiere")
    
    with col2:
        def clear_text():
            st.session_state.user_input = ""
        st.button("Lösche Input", on_click=clear_text)


    if analyze_clicked:
        if user_input.strip():
            try:
                # Send POST request to Flask API
                response = requests.post(
                    "http://localhost:5000/predict_leaning",
                    json={"text": user_input}
                )
                if response.status_code == 200:
                    st.session_state.analysis_result = response.json()
                    st.success("Analyse abgeschlossen!")

                else:
                    st.error(f"Fehler vom Server: {response.status_code}")
            except Exception as e:
                st.error(f"Fehler beim Verbinden mit der API: {e}")
        else:
            st.warning("Bitte gib einen Text ein.")

    
    # Display the analysis result if available
    if st.session_state.analysis_result:
        st.subheader("Analyseergebnis:")

        # Convert to percentages
        result_percent = {k: v * 100 for k, v in st.session_state.analysis_result.items()}

        # Define the desired order
        order = ["left", "left-center", "center", "right-center", "right"]
        values = [result_percent[k] for k in order]

        # Create bar chart using Plotly
        fig = go.Figure(data=[
            go.Bar(
                name='Politische Neigung',
                x=order,
                y=values,
                marker_color=colors_classes
            )
        ])
        fig.update_layout(
            yaxis_title="Prozent",
            title="Politische Neigung (in %)",
            yaxis=dict(range=[0, 100]),
            plot_bgcolor="rgba(240, 240, 240, 1)",
            paper_bgcolor="rgba(240, 240, 240, 1)"

        )

        # Show chart
        st.plotly_chart(fig)


        # Determine the dominant political leaning
        dominant_leaning = max(result_percent, key=result_percent.get)
        st.write(f"Der Text ist am ehesten **{dominant_leaning}** gerichtet.")

    # whitespace
    st.text("")
    st.text("")
    st.write("---")

    st.title("Politische Neigung versch. Medien")
    st.text("")

    chart_col, nav_col = st.columns([11, 1])

    with chart_col:
        if st.session_state.current_chart_display == "chart1":
            fig = create_stacked_chart(df=df_media, chart_title="nach Klassendurchschnitt")
            st.plotly_chart(fig, use_container_width=True)
        elif st.session_state.current_chart_display == "chart2":
            fig2 = create_stacked_chart(df=df_article, chart_title="nach Anzahl Artikel")
            st.plotly_chart(fig2, use_container_width=True)

    with nav_col:
        st.markdown("<div style='height: 210px;'></div>", unsafe_allow_html=True)
        if st.session_state.current_chart_display == "chart1":
            if st.button("->", key="next_chart_btn"):
                st.session_state.current_chart_display = "chart2"
                st.rerun()
        elif st.session_state.current_chart_display == "chart2":
            if st.button("<-", key="prev_chart_btn"):
                st.session_state.current_chart_display = "chart1"
                st.rerun()