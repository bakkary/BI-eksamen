# exam.py
import streamlit as st
import Graphs  # This imports the graphs module you've created

# Function to display the homepage content
def show_homepage():
    st.title('Homepage')
    st.write("Welcome to the Streamlit App!")

# Main function that runs the app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Homepage", "Generate Graphs"])

    if page == "Homepage":
        show_homepage()
    elif page == "Generate Graphs":
        Graphs.show_graphs()  # This calls a function from graphs.py to display its content

if __name__ == "__main__":
    main()
