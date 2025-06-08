# ---- Setup (beginning) ----

import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
from pathlib import Path
import ast

# Set page configuration
st.set_page_config(
    page_title="Politische Texterkennung",
    layout="wide",
    initial_sidebar_state="auto"
)

# Custom CSS for the text box
st.markdown(
    """
    <style>

    /* Style for the text input box */
    .stTextArea > div > div > textarea {
        border: 1px solid #cccccc;
        border-radius: 5px;
    }

    /* Target individual columns created by st.columns */
    [data-testid="stColumn"] {
        padding: 0px !important; /* Removes padding from inside the column */
        margin: 0px !important;  /* Removes margin from around the column */
    }

    /* Target the container that holds the columns (the "row" created by st.columns) */
    [data-testid="stHorizontalBlock"] {
        padding: 0px !important; /* Removes padding from the row container */
        margin: 0px !important;  /* Removes margin from the row container */
        gap: 0px !important;     /* Removes gap between columns in flexbox layouts */
    }

    /* Button colours */

    /* Primary button styling */

    button[kind="primary"] {
        background-color: #4CAF50;
        color: white;        
        border: none;
        outline: none;
    }
    /* Hover state */
    button[kind="primary"]:hover {
        background-color: #45a049;
    }
    /* Active (clicked/held down) state */
    button[kind="primary"]:active {
        background-color: #3e8e41;
    }
    /* Focus state (after click or tabbing) */
    button[kind="primary"]:focus,
    button[kind="primary"]:focus-visible {
        background-color: #4CAF50;
        color: white;
        border: none;
        outline: none;
    }

    /* Secondary button styling */

    button[kind="secondary"] {
        background-color: #CC3333;
        color: white !important;
        border: none;
        outline: none;
    }
    /* Hover state */
    button[kind="secondary"]:hover {
        background-color: #990000;
        color: white !important;
    }
    /* Active (clicked/held down) state */
    button[kind="secondary"]:active {
        background-color: #800000;
        color: white !important;
    }
    /* Focus state */
    button[kind="secondary"]:focus,
    button[kind="secondary"]:focus-visible {
        background-color: #CC3333;
        color: white !important;
        border: none;
        outline: none;
        box-shadow: none;
    }

    /* Tertiary button styling */

    button[kind="tertiary"] {
        background-color: #e0e0e0;
        color: #333333 !important;
        border: none;
        outline: none;
        padding: 0px 10px !important;
    /* Hover state */
    button[kind="tertiary"]:hover {
        background-color: #d0d0d0;
        color: #333333 !important;
    }
    /* Active (clicked/held down) state */
    button[kind="tertiary"]:active {
        background-color: #c0c0c0;
        color: #333333 !important;
    }
    /* Focus state */
    button[kind="tertiary"]:focus,
    button[kind="tertiary"]:focus-visible {
        background-color: #e0e0e0;
        color: #333333 !important;
        border: none;
        outline: none;
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

# ---- Setup (end) ----


# ---- Colour section (beginning) ----
    
# Background colour of the charts. If you want it to be not white, change it here
background_colour = "#ffffff"

# Colours of the classes. Edit the list to change the colours.
colors_classes = [
    "#4169E1",
    "#8195D1",
    "#C0C0C0",
    "#C67A7A",
    "#CC3333",
]

color_map_classes = {
    "left": colors_classes[0],    
    "left-center": colors_classes[1],  
    "center": colors_classes[2], 
    "right-center": colors_classes[3],
    "right": colors_classes[4]    
}

# ---- Colour section (end) ----


# ---- Data and data preperation (beginning) ----

# Translate the classes
class_translations = {
    "left": "links",
    "left-center": "links-Mitte",
    "center": "Mitte",
    "right-center": "rechts-Mitte",
    "right": "rechts"
}

# Data Prep for article-based evaluation and to correctly display order
# Constructing the absolute path because relative didn't work

script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent
article_data_path = project_root / "article_classification" / "formatted_articles.csv"
article_data_path_str = str(article_data_path)
df_csv = pd.read_csv(article_data_path_str, sep=";")

classes_order = ["left", "left-center", "center", "right-center", "right"]
media_order = df_csv["Medium"].unique()
media_order = sorted(media_order.tolist())

df_csv['Medium'] = pd.Categorical(df_csv['Medium'], categories=media_order, ordered=True)
df_csv['Class'] = pd.Categorical(df_csv['Class'], categories=classes_order, ordered=True)

df_csv = df_csv.sort_values(by=['Medium', 'Class'])
df_csv = df_csv.reset_index(drop=True)

df_csv["Class_german"] = df_csv["Class"].map(class_translations)

# For class-based evaluation
medium_data_path = project_root / "article_classification" / "formatted_media.csv"
medium_data_path_str = str(medium_data_path)
medium_df = pd.read_csv(medium_data_path_str, sep=";")

def parse_string_to_list(s):
    parsed_obj = ast.literal_eval(s)
    return parsed_obj

medium_df['probabilities'] = medium_df['probabilities'].apply(parse_string_to_list)

data_class_avg = dict(zip(medium_df['Medium'], medium_df['probabilities']))
print(data_class_avg)

# Convert dictionary to dataframe for plotting
def dict_to_pdf(data_dict):
    """
    Converts a dictionary of media data into a Pandas DataFrame suitable for Plotly bar charts.

    Args:
        data_dict (dict): A dictionary where:
            - Keys are strings representing media names (e.g., "Tagesschau", "Bild").
            - Values are lists of floats, where each float corresponds to a political class's
              proportion/score. The order of these floats must align with the `classes_order`
              list.

    Returns:
        data_df (pandas.DataFrame): A DataFrame with the following columns:
            - 'Medium' (str): The name of the media outlet.
            - 'Class' (str): The political leaning class (e.g., "left", "center").
            - 'Value' (float): The numerical proportion/score multiplied by 100 (as a percentage).
            - 'Percentage' (str): The 'Value' formatted as a string with one decimal place and a '%' sign.
    """

    dataframe_rows = []
    for medium, values_array in data_dict.items():
        for i, value in enumerate(values_array):
            row = {
                "Medium": medium,
                "Class": classes_order[i],
                "Value": value*100
            }
            dataframe_rows.append(row)
    data_df = pd.DataFrame(dataframe_rows)
    data_df["Percentage"] = data_df["Value"].apply(lambda x: f"{x:.1f}%")

    return data_df

df_media = dict_to_pdf(data_class_avg)
df_media["Class_german"] = df_media["Class"].map(class_translations)

# ---- Data and data preperation (end) ----


# ---- Other global functions (beginning) ----

# Make a stacked bar-chart for the media-analysis
def create_stacked_chart(*, df, chart_title):
    """
    Creates a stacked bar-chart for the media-analysis, not the user input.

    Args:
    df (pandas.DataFrame): The input DataFrame containing the data for the stacked bar chart.
                          It must include the following columns:
                          - 'Value' (numeric): The value for the x-axis (e.g., percentage score).
                          - 'Medium' (str): The categorical data for the y-axis (e.g., media outlet names).
                          - 'Class' (str): The categorical data for coloring and stacking (e.g., political leanings).
                          - 'Percentage' (str): The text to be displayed inside the bars (e.g., "XX.X%").
    chart_title (str): The title to be displayed above the chart.

    Returns:
        plotly.graph_objects.Figure: A Plotly Figure object representing the configured
                                     horizontal stacked bar chart.
    """
    
    fig = px.bar(
        df,
        title=chart_title,
        x="Value",
        y="Medium",
        color="Class",
        orientation="h",
        barmode="stack",
        text="Percentage",
        color_discrete_map=color_map_classes,
        category_orders={"Medium": media_order, "Class": classes_order},
        custom_data=["Class_german"]
    )

    fig.update_layout(
        xaxis_visible=False,
        xaxis_showticklabels=False,
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False,
        yaxis_title=None,
        paper_bgcolor=background_colour,
        plot_bgcolor=background_colour,
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
              "<b>Neigung:</b> %{customdata[0]}<br>" +
              "<b>Prozent:</b> %{text}" +
              "<extra></extra>"
    )
    return fig

def format_links_with_bullets(html_string):
    """
    Transforms a string of <a href> links separated by <br> tags
    into an unordered HTML list with bullet points.

    Args:
    html_string (str): A string containing one or more HTML anchor (`<a>`) tags,
                       separated by `<br>` tags, representing a list of links.
                       Example: "<a href='url1'>Link1</a><br><a href='url2'>Link2</a>"

    Returns:
        str: An HTML string representing an unordered list (`<ul>`) where each original
             link is formatted as a list item (`<li>`). Returns an empty string if the
             input `html_string` is empty or contains only whitespace.
    """

    if not html_string or html_string.strip() == "":
        return ""

    individual_links = html_string.split('<br>')

    # Filter out any empty strings that might result from splitting (e.g., if there's a trailing <br>)
    # and wrap each non-empty link in an <li> tag
    list_items = [f"<li>{link.strip()}</li>" for link in individual_links if link.strip()]

    # Join the list items and wrap them in a <ul> tag
    bulleted_html = f"<ul>{''.join(list_items)}</ul>"

    return bulleted_html

# ---- Other global functions (end) ----


# Title of the application
st.title("Bestimmung der politischen Neigung von Texten")

# whitespace
st.text("")


# ---- Analysis of user input (beginning) ----

st.header("Dein eigener Text")

# whitespace
st.text("")

# Layout
col_user_input, col_spacer, col_analysis = st.columns([36, 2, 40])

# Section of user input, i. e. the text box and its buttons
with col_user_input:

    # Text box for user input
    user_input = st.text_area(
        "Gib einen Text ein:",
        height=410,
        placeholder="Dein politischer Text...",
        key="user_input"
    )

    # Buttons
    col1, col2 = st.columns([5, 1])

    # Triggers analysis
    with col1:
        analyze_clicked = st.button("Analysiere", type="primary")
    
    # Deletes user input and only user input. Rest of the page remains uneffected
    with col2:
        def clear_text():
            st.session_state.user_input = ""
        st.button("Lösche Input", on_click=clear_text, type="secondary")

# Displays the analysis of the user input
with col_analysis:
    st.subheader("Politische Neigung (in %)")

    # API call
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
                else:
                    st.error(f"Fehler vom Server: {response.status_code}")
            except Exception as e:
                st.error(f"Fehler beim Verbinden mit der API: {e}")
        else:
            st.warning("Bitte gib einen Text ein.")

    
    
    # Display the analysis result if API successful
    if st.session_state.analysis_result:

        # Convert to percentages since API returns fractions like 0.1
        result_percent = {k: v * 100 for k, v in st.session_state.analysis_result.items()}

        # Sort
        values = [result_percent[k] for k in classes_order]

        # Translate
        classes_order_translated = [class_translations[word] for word in classes_order]

        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=classes_order_translated,
                y=values,
                marker_color=colors_classes
            )
        ])
        fig.update_layout(
            yaxis=dict(range=[0, 100]),
            yaxis_title=None,
            height=400,
            margin=dict(l=0, r=0, t=0, b=0),
            plot_bgcolor=background_colour,
            paper_bgcolor=background_colour
        )

        # Show chart
        st.plotly_chart(fig)

        # Determine the dominant political leaning
        dominant_leaning = max(result_percent, key=result_percent.get)
        dominant_leaning = class_translations.get(dominant_leaning, dominant_leaning)
        st.write(f"Der Text ist am ehesten **{dominant_leaning}** gerichtet.")

# ---- Analysis of user input (end) ----
        

# whitespace
st.text("")
st.text("")
st.write("---")


# ---- Analysis of different media (beginning) ----

st.header("aus Medien")
st.text("")
st.text("")

# Layout for chart and nav-button
chart_col, nav_col = st.columns([30, 1])

# Charts
with chart_col:
    
    # Chart for media analysis based on class-average
    if st.session_state.current_chart_display == "chart1":
        fig = create_stacked_chart(df=df_media, chart_title="nach Klassendurchschnitt")
        st.plotly_chart(fig, use_container_width=True)

    # Chart for media analysis based on class of its articles
    elif st.session_state.current_chart_display == "chart2":

        # Layout for chart and articles
        article_chart_col, article_col = st.columns([3, 1])

        with article_chart_col:
            fig2 = create_stacked_chart(df=df_csv, chart_title="nach Anzahl Artikel")
            st.plotly_chart(fig2, use_container_width=True, on_select="rerun", key="plotly_selection_data")

        # ---- Enable clicking on the chart (beginning) ----
            
        with article_col:
            if st.session_state.plotly_selection_data and \
            st.session_state.plotly_selection_data.get('selection') and \
            st.session_state.plotly_selection_data['selection'].get('points'):

                clicked_point = st.session_state.plotly_selection_data['selection']['points'][0]

                # Extract medium
                medium = clicked_point.get('y') # The y-axis value, which is 'Medium'

                # Extract class
                curve_number = clicked_point.get('curve_number') # Index of the trace (which is 'Class')
                chart_class = "Unknown Class" # Default in case curve_number is invalid
                if curve_number is not None and curve_number < len(fig2.data):
                    chart_class = fig2.data[curve_number].name
                else:
                    st.warning(f"Could not determine 'Class' name. curveNumber: {curve_number} is out of range or missing.")
                german_class = class_translations.get(chart_class, chart_class)

                # Extract article links and format them
                article_links = df_csv["clickable_title"][
                    (df_csv["Medium"] == medium) & (df_csv["Class"] == chart_class)
                ]
                article_links = article_links.iloc[0]
                bullet_links = format_links_with_bullets(article_links)

                # Display results
                st.success(f'**Artikel aus "{medium}" mit der Neigung "{german_class}"**')
                st.markdown(bullet_links, unsafe_allow_html=True)
            else:
                # This will be shown when no segment has been clicked yet, or if the selection is cleared
                st.info("Klicke auf ein farbiges Segment eines Balkens, um zu sehen, aus welchen Artikeln es besteht (klicke mehrmals, wenn es beim ersten Klick nicht selektiert).")

        # ---- Enable clicking on the chart (end) ----
            
# ---- Analysis of different media (end) ----
            

# ---- Navigation button (beginning) ----

with nav_col:
    st.markdown("<div style='height: 210px';></div>", unsafe_allow_html=True)
    if st.session_state.current_chart_display == "chart1":
        if st.button("->", key="next_chart_btn", type="tertiary"):
            st.session_state.current_chart_display = "chart2"
            st.rerun()
    elif st.session_state.current_chart_display == "chart2":
        if st.button("<-", key="prev_chart_btn", type="tertiary"):
            st.session_state.current_chart_display = "chart1"
            st.rerun()

# ---- Navigation button (end) ----