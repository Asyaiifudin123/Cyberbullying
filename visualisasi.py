import pickle
import streamlit as st
import streamlit.components.v1 as stc
import pandas as pd
import plotly.express as px


def visualisasi():

    eda = ['Exploratory Data Analysis', 'CyberBullying Word', 'Gender Cyberbullying Words',
           'Age Cyberbullying Words', 'Ethnicity Cyberbullying Words', 'Religion Cyberbullying Words']
    choice = st.sidebar.selectbox("Exploratory Data Analysis", eda)

    if choice == 'Exploratory Data Analysis':
        st.subheader("Cyberbullying Sentiment Classification")
        value_counts()
    elif choice == "CyberBullying Word":
        st.subheader("Top 10 CyberBullying Word")
        top_bullying()
    elif choice == "Gender Cyberbullying Words":
        st.subheader("Top 10 Gender Cyberbullying Words")
        gender_bullying()
    elif choice == "Age Cyberbullying Words":
        st.subheader("Top 10 Age Cyberbullying Words")
        age_bullying()
    elif choice == "Ethnicity Cyberbullying Words":
        st.subheader("Top 10 Ethnicity Cyberbullying Words")
        ethnicity_bullying()
    elif choice == "Religion Cyberbullying Words":
        st.subheader("Top 10 Religion Cyberbullying Words")
        religion_bullying()


def value_counts():
    # Load DataFrame from the pickle file
    with open('value_counts.pkl', 'rb') as file:
        loaded_value_counts = pickle.load(file)

    # Convert Series to DataFrame if needed
    if isinstance(loaded_value_counts, pd.Series):
        loaded_value_counts = loaded_value_counts.to_frame(
            name='Count').reset_index()

    # Display the loaded value counts using Streamlit with custom styling
    st.table(
        loaded_value_counts
        .style
        .apply(lambda x: ['background: #fffffff; color: white' if x.name == 'Category' else '' for i in x],
               axis=1)
        # Cell text alignment and font size
        .set_properties(**{'text-align': 'center', 'font-size': '16px'})
        # Cell background gradient
        .background_gradient(cmap='Blues', subset=['Count'])
    )

    st.subheader("Chart Cyberbullying Sentiment ")

    # Display the pie chart using Plotly Express with default color
    fig = px.pie(loaded_value_counts, values='Count', names='sentiment',
                 title=' ',
                 width=700, height=500,
                 labels={'Count': 'Value Counts'},
                 hover_data=['Count'],  # Add hover data to show values
                 hole=0.4,  # Set the size of the hole in the center of the pie chart
                 template='plotly_dark',  # Set the template to a dark theme
                 )

    # Set the shape of the pie chart to "star"
    fig.update_traces(marker=dict(line=dict(color='#2e4053', width=2)), pull=[
                      0.1, 0.1, 0.1, 0.1, 0.1])

    # Update the layout for better appearance
    fig.update_layout(
        title_x=0.5,  # Center the title
        font=dict(size=14),
    )

    # Display the Plotly Express figure using Streamlit
    st.plotly_chart(fig)


def top_bullying():
    # Load DataFrame from the pickle file
    with open('tweet_list1.pkl', 'rb') as file:
        loaded_tweet_list = pickle.load(file)

    st.table(
        loaded_tweet_list.style
        .set_table_styles([
            # Header background color
            {'selector': 'th', 'props': [('background-color', '#fffffff')]},
            # Header text alignment and color
            {'selector': 'th', 'props': [
                ('text-align', 'center'), ('color', 'white')]},
            # Cell text alignment and font size
            {'selector': 'td', 'props': [
                ('text-align', 'center'), ('font-size', '16px')]},
        ])
        .bar(subset=['Count'], color='#1f77b4', vmin=0, vmax=loaded_tweet_list['Count'].max())
        .background_gradient(cmap='Blues', subset=['Count'])
    )

    st.subheader("Chart Top 10 Bullying Word")

    # Sort the DataFrame by 'Count' in descending order
    loaded_tweet_list = loaded_tweet_list.sort_values(
        by='Count', ascending=True)

    # Display the bar chart using Plotly Express with colorful bars
    fig = px.bar(loaded_tweet_list, x="Count", y="Words", title=' ',
                 labels={'Words': 'Top 10 Bullying Words', 'Count': 'Count'},
                 orientation='h', width=700, height=700, color='Count', color_continuous_scale='Blues')

    # Update the layout for better appearance
    fig.update_layout(
        title_x=0.5,  # Center the title
        xaxis_title='Count',
        yaxis_title='Top 10 Bullying Words',
        font=dict(size=14),
        coloraxis_colorbar=dict(title='Count')
    )

    # Display the Plotly Express figure using Streamlit
    st.plotly_chart(fig)


def gender_bullying():
    # Load DataFrame from the pickle file
    with open('tweet_list2.pkl', 'rb') as file:
        loaded_tweet_list2 = pickle.load(file)

    st.table(
        loaded_tweet_list2.style
        .set_table_styles([
            # Header background color
            {'selector': 'th', 'props': [('background-color', '#fffffff')]},
            # Header text alignment and color
            {'selector': 'th', 'props': [
                ('text-align', 'center'), ('color', 'white')]},
            # Cell text alignment and font size
            {'selector': 'td', 'props': [
                ('text-align', 'center'), ('font-size', '16px')]},
        ])
        .bar(subset=['Count'], color='#1f77b4', vmin=0, vmax=loaded_tweet_list2['Count'].max())
        .background_gradient(cmap='Blues', subset=['Count'])
    )

    st.subheader("Chart Top 10 Gender CyberBullying Word")

    # Sort the DataFrame by 'Count' in descending order
    loaded_tweet_list2 = loaded_tweet_list2.sort_values(
        by='Count', ascending=True)

    # Display the bar chart using Plotly Express with colorful bars
    fig = px.bar(loaded_tweet_list2, x="Count", y="Top of Words", title=' ',
                 labels={'Words': 'Top 10 Gender Cyberbullying Words',
                         'Count': 'Count'},
                 orientation='h', width=700, height=700, color='Count', color_continuous_scale='Blues')

    # Update the layout for better appearance
    fig.update_layout(
        title_x=0.5,  # Center the title
        xaxis_title='Count',
        yaxis_title='Top 10 Gender Cyberbullying Words',
        font=dict(size=14),
        coloraxis_colorbar=dict(title='Count')
    )

    # Display the Plotly Express figure using Streamlit
    st.plotly_chart(fig)


def age_bullying():
    # Load DataFrame from the pickle file
    with open('tweet_list3.pkl', 'rb') as file:
        loaded_tweet_list3 = pickle.load(file)

    st.table(
        loaded_tweet_list3.style
        .set_table_styles([
            # Header background color
            {'selector': 'th', 'props': [('background-color', '#fffffff')]},
            # Header text alignment and color
            {'selector': 'th', 'props': [
                ('text-align', 'center'), ('color', 'white')]},
            # Cell text alignment and font size
            {'selector': 'td', 'props': [
                ('text-align', 'center'), ('font-size', '16px')]},
        ])
        .bar(subset=['Count'], color='#1f77b4', vmin=0, vmax=loaded_tweet_list3['Count'].max())
        .background_gradient(cmap='Blues', subset=['Count'])
    )

    st.subheader("Chart Top 10 Age CyberBullying Word")

    # Sort the DataFrame by 'Count' in descending order
    loaded_tweet_list3 = loaded_tweet_list3.sort_values(
        by='Count', ascending=True)

    # Display the bar chart using Plotly Express with colorful bars
    fig = px.bar(loaded_tweet_list3, x="Count", y="Top of Words", title=' ',
                 labels={'Words': 'Top 10 Age Cyberbullying Words',
                         'Count': 'Count'},
                 orientation='h', width=700, height=700, color='Count', color_continuous_scale='Blues')

    # Update the layout for better appearance
    fig.update_layout(
        title_x=0.5,  # Center the title
        xaxis_title='Count',
        yaxis_title='Top 10 Age Cyberbullying Words',
        font=dict(size=14),
        coloraxis_colorbar=dict(title='Count')
    )

    # Display the Plotly Express figure using Streamlit
    st.plotly_chart(fig)


def ethnicity_bullying():
    # Load DataFrame from the pickle file
    with open('tweet_list4.pkl', 'rb') as file:
        loaded_tweet_list4 = pickle.load(file)

    st.table(
        loaded_tweet_list4.style
        .set_table_styles([
            # Header background color
            {'selector': 'th', 'props': [('background-color', '#fffffff')]},
            # Header text alignment and color
            {'selector': 'th', 'props': [
                ('text-align', 'center'), ('color', 'white')]},
            # Cell text alignment and font size
            {'selector': 'td', 'props': [
                ('text-align', 'center'), ('font-size', '16px')]},
        ])
        .bar(subset=['Count'], color='#1f77b4', vmin=0, vmax=loaded_tweet_list4['Count'].max())
        .background_gradient(cmap='Blues', subset=['Count'])
    )

    st.subheader("Chart Top 10 Ethnicity CyberBullying Word")

    # Sort the DataFrame by 'Count' in descending order
    loaded_tweet_list4 = loaded_tweet_list4.sort_values(
        by='Count', ascending=True)

    # Display the bar chart using Plotly Express with colorful bars
    fig = px.bar(loaded_tweet_list4, x="Count", y="Top of Words", title=' ',
                 labels={'Words': 'Top 10 Ethnicity Cyberbullying Words',
                         'Count': 'Count'},
                 orientation='h', width=700, height=700, color='Count', color_continuous_scale='Blues')

    # Update the layout for better appearance
    fig.update_layout(
        title_x=0.5,  # Center the title
        xaxis_title='Count',
        yaxis_title='Top 10 Ethnicity Cyberbullying Words',
        font=dict(size=14),
        coloraxis_colorbar=dict(title='Count')
    )

    # Display the Plotly Express figure using Streamlit
    st.plotly_chart(fig)


def religion_bullying():
    # Load DataFrame from the pickle file
    with open('tweet_list5.pkl', 'rb') as file:
        loaded_tweet_list5 = pickle.load(file)

    st.table(
        loaded_tweet_list5.style
        .set_table_styles([
            # Header background color
            {'selector': 'th', 'props': [('background-color', '#fffffff')]},
            # Header text alignment and color
            {'selector': 'th', 'props': [
                ('text-align', 'center'), ('color', 'white')]},
            # Cell text alignment and font size
            {'selector': 'td', 'props': [
                ('text-align', 'center'), ('font-size', '16px')]},
        ])
        .bar(subset=['Count'], color='#1f77b4', vmin=0, vmax=loaded_tweet_list5['Count'].max())
        .background_gradient(cmap='Blues', subset=['Count'])
    )

    st.subheader("Chart Top 10 Religion CyberBullying Word")

    # Sort the DataFrame by 'Count' in descending order
    loaded_tweet_list5 = loaded_tweet_list5.sort_values(
        by='Count', ascending=True)

    # Display the bar chart using Plotly Express with colorful bars
    fig = px.bar(loaded_tweet_list5, x="Count", y="Top of Words", title=' ',
                 labels={'Words': 'Top 10 Religion Cyberbullying Words',
                         'Count': 'Count'},
                 orientation='h', width=700, height=700, color='Count', color_continuous_scale='Blues')

    # Update the layout for better appearance
    fig.update_layout(
        title_x=0.5,  # Center the title
        xaxis_title='Count',
        yaxis_title='Top 10 Religion Cyberbullying Words',
        font=dict(size=14),
        coloraxis_colorbar=dict(title='Count')
    )

    # Display the Plotly Express figure using Streamlit
    st.plotly_chart(fig)


if __name__ == '__main__':
    visualisasi()
