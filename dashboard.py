import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Data Visualization Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
    .plot-container {
        border: 2px solid #e1e5e9;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'plots' not in st.session_state:
    st.session_state.plots = []

def load_data(uploaded_file):
    """Load and validate CSV data"""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def get_numeric_columns(df):
    """Get numeric columns from dataframe"""
    return df.select_dtypes(include=[np.number]).columns.tolist()

def get_categorical_columns(df):
    """Get categorical columns from dataframe"""
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

def create_plot(plot_type, df, x_col, y_col=None, hue_col=None, style='default'):
    """Create different types of plots"""
    # Set style
    plt.style.use(style if style != 'default' else 'default')
    sns.set_style("whitegrid" if style == 'default' else style)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    try:
        if plot_type == "Histogram":
            if hue_col and hue_col != "None":
                sns.histplot(data=df, x=x_col, hue=hue_col, ax=ax, kde=True)
            else:
                sns.histplot(data=df, x=x_col, ax=ax, kde=True)
            ax.set_title(f'Histogram of {x_col}')
            
        elif plot_type == "Scatter Plot":
            if y_col and hue_col and hue_col != "None":
                sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax)
            elif y_col:
                sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
            ax.set_title(f'Scatter Plot: {x_col} vs {y_col}')
            
        elif plot_type == "Line Plot":
            if y_col and hue_col and hue_col != "None":
                sns.lineplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax)
            elif y_col:
                sns.lineplot(data=df, x=x_col, y=y_col, ax=ax)
            ax.set_title(f'Line Plot: {x_col} vs {y_col}')
            
        elif plot_type == "Bar Plot":
            if y_col and hue_col and hue_col != "None":
                sns.barplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax)
            elif y_col:
                sns.barplot(data=df, x=x_col, y=y_col, ax=ax)
            else:
                df[x_col].value_counts().plot(kind='bar', ax=ax)
            ax.set_title(f'Bar Plot of {x_col}')
            plt.xticks(rotation=45)
            
        elif plot_type == "Box Plot":
            if y_col and hue_col and hue_col != "None":
                sns.boxplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax)
            elif y_col:
                sns.boxplot(data=df, x=x_col, y=y_col, ax=ax)
            else:
                sns.boxplot(data=df, y=x_col, ax=ax)
            ax.set_title(f'Box Plot of {x_col}')
            
        elif plot_type == "Violin Plot":
            if y_col and hue_col and hue_col != "None":
                sns.violinplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax)
            elif y_col:
                sns.violinplot(data=df, x=x_col, y=y_col, ax=ax)
            else:
                sns.violinplot(data=df, y=x_col, ax=ax)
            ax.set_title(f'Violin Plot of {x_col}')
            
        elif plot_type == "Heatmap":
            numeric_cols = get_numeric_columns(df)
            if len(numeric_cols) > 1:
                correlation_matrix = df[numeric_cols].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
                ax.set_title('Correlation Heatmap')
            else:
                ax.text(0.5, 0.5, 'Need at least 2 numeric columns for heatmap', 
                       ha='center', va='center', transform=ax.transAxes)
                
        elif plot_type == "Pair Plot":
            numeric_cols = get_numeric_columns(df)
            if len(numeric_cols) > 1:
                # For pair plot, we need to create it differently
                plt.close(fig)
                if hue_col and hue_col != "None":
                    pair_plot = sns.pairplot(df[numeric_cols + [hue_col]], hue=hue_col)
                else:
                    pair_plot = sns.pairplot(df[numeric_cols])
                return pair_plot.fig
            else:
                ax.text(0.5, 0.5, 'Need at least 2 numeric columns for pair plot', 
                       ha='center', va='center', transform=ax.transAxes)
                
        elif plot_type == "Count Plot":
            if hue_col and hue_col != "None":
                sns.countplot(data=df, x=x_col, hue=hue_col, ax=ax)
            else:
                sns.countplot(data=df, x=x_col, ax=ax)
            ax.set_title(f'Count Plot of {x_col}')
            plt.xticks(rotation=45)
            
        elif plot_type == "Distribution Plot":
            sns.distplot(df[x_col], ax=ax)
            ax.set_title(f'Distribution Plot of {x_col}')
            
        plt.tight_layout()
        return fig
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Error creating plot: {str(e)}', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Plot Error')
        return fig

def download_plot(fig, plot_name):
    """Create download link for plot"""
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    
    b64_img = base64.b64encode(img_buffer.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{b64_img}" download="{plot_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png">Download Plot</a>'
    return href

def main():
    # Main header
    st.markdown('<h1 class="main-header">📊 Data Visualization Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar for file upload and options
    with st.sidebar:
        st.markdown('<h2 class="sub-header">🔧 Configuration</h2>', unsafe_allow_html=True)
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=['csv'],
            help="Upload your CSV file to start creating visualizations"
        )
        
        if uploaded_file is not None:
            # Load data
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state.data = df
                st.success(f"✅ File loaded successfully!")
                st.info(f"📋 Shape: {df.shape[0]} rows, {df.shape[1]} columns")
                
                # Display basic info
                with st.expander("📊 Dataset Preview"):
                    st.dataframe(df.head())
                
                with st.expander("ℹ️ Dataset Info"):
                    st.write("**Columns:**", list(df.columns))
                    st.write("**Data Types:**")
                    st.write(df.dtypes)
                    st.write("**Missing Values:**")
                    st.write(df.isnull().sum())
    
    # Main content area
    if st.session_state.data is not None:
        df = st.session_state.data
        
        # Plot configuration
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<h2 class="sub-header">🎨 Plot Configuration</h2>', unsafe_allow_html=True)
            
            # Plot type selection
            plot_types = [
                "Histogram", "Scatter Plot", "Line Plot", "Bar Plot", 
                "Box Plot", "Violin Plot", "Heatmap", "Pair Plot", 
                "Count Plot", "Distribution Plot"
            ]
            
            plot_type = st.selectbox("Select Plot Type", plot_types)
            
            # Column selection based on plot type
            numeric_cols = get_numeric_columns(df)
            categorical_cols = get_categorical_columns(df)
            all_cols = list(df.columns)
            
            if plot_type in ["Histogram", "Distribution Plot", "Box Plot", "Violin Plot"]:
                x_col = st.selectbox("Select Column", numeric_cols)
                y_col = None
            elif plot_type in ["Scatter Plot", "Line Plot", "Bar Plot"]:
                x_col = st.selectbox("Select X Column", all_cols)
                y_col = st.selectbox("Select Y Column", numeric_cols)
            elif plot_type == "Count Plot":
                x_col = st.selectbox("Select Column", categorical_cols + numeric_cols)
                y_col = None
            elif plot_type in ["Heatmap", "Pair Plot"]:
                x_col = None
                y_col = None
            else:
                x_col = st.selectbox("Select X Column", all_cols)
                y_col = st.selectbox("Select Y Column", all_cols)
            
            # Hue column (for grouping)
            hue_options = ["None"] + categorical_cols
            hue_col = st.selectbox("Select Hue Column (Optional)", hue_options)
            
        with col2:
            st.markdown('<h2 class="sub-header">🎭 Style Configuration</h2>', unsafe_allow_html=True)
            
            # Style selection
            styles = ['default', 'darkgrid', 'whitegrid', 'dark', 'white', 'seaborn']
            style = st.selectbox("Select Plot Style", styles)
            
            # Color palette
            palettes = ['deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind']
            palette = st.selectbox("Select Color Palette", palettes)
            
            # Set the palette
            sns.set_palette(palette)
        
        # Generate plot button
        if st.button("🎯 Generate Plot", type="primary"):
            if plot_type in ["Heatmap", "Pair Plot"] or x_col is not None:
                with st.spinner("Creating your visualization..."):
                    fig = create_plot(plot_type, df, x_col, y_col, hue_col, style)
                    
                    if fig is not None:
                        st.session_state.plots.append({
                            'figure': fig,
                            'name': f"{plot_type}_{datetime.now().strftime('%H%M%S')}",
                            'type': plot_type,
                            'x_col': x_col,
                            'y_col': y_col,
                            'hue_col': hue_col,
                            'style': style
                        })
            else:
                st.error("Please select required columns for the plot type.")
        
        # Display generated plots
        if st.session_state.plots:
            st.markdown('<h2 class="sub-header">📈 Generated Visualizations</h2>', unsafe_allow_html=True)
            
            # Plot management
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                plot_indices = list(range(len(st.session_state.plots)))
                plot_names = [f"{i+1}. {plot['type']} - {plot['name']}" for i, plot in enumerate(st.session_state.plots)]
                selected_plot = st.selectbox("Select Plot to View/Download", plot_names)
                
            with col2:
                if st.button("🗑️ Clear All Plots"):
                    st.session_state.plots = []
                    st.rerun()
                    
            with col3:
                if st.button("🔄 Refresh"):
                    st.rerun()
            
            # Display selected plot
            if selected_plot:
                plot_index = int(selected_plot.split('.')[0]) - 1
                selected_plot_data = st.session_state.plots[plot_index]
                
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                
                # Plot info
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Plot Type:** {selected_plot_data['type']}")
                    if selected_plot_data['x_col']:
                        st.write(f"**X Column:** {selected_plot_data['x_col']}")
                    if selected_plot_data['y_col']:
                        st.write(f"**Y Column:** {selected_plot_data['y_col']}")
                    if selected_plot_data['hue_col'] and selected_plot_data['hue_col'] != "None":
                        st.write(f"**Hue Column:** {selected_plot_data['hue_col']}")
                    st.write(f"**Style:** {selected_plot_data['style']}")
                
                with col2:
                    # Download button
                    download_link = download_plot(selected_plot_data['figure'], selected_plot_data['name'])
                    st.markdown(f"### {download_link}", unsafe_allow_html=True)
                
                # Display plot
                st.pyplot(selected_plot_data['figure'])
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Statistics section
        with st.expander("📊 Dataset Statistics"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Descriptive Statistics:**")
                st.dataframe(df.describe())
                
            with col2:
                st.write("**Missing Values:**")
                missing_data = df.isnull().sum()
                st.dataframe(missing_data[missing_data > 0])
                
                if len(numeric_cols) > 0:
                    st.write("**Correlation Matrix:**")
                    st.dataframe(df[numeric_cols].corr())
    
    else:
        # Welcome message
        st.markdown("""
        ## 👋 Welcome to the Data Visualization Dashboard!
        
        This dashboard allows you to:
        - 📁 Upload CSV files
        - 📊 Generate various types of plots and graphs
        - 🎨 Customize plot styles and colors
        - 💾 Download your visualizations
        
        ### 🚀 Getting Started:
        1. Upload your CSV file using the sidebar
        2. Select your preferred plot type and columns
        3. Customize the style and generate your visualization
        4. Download your plots for use in presentations or reports
        
        ### 📊 Supported Plot Types:
        - Histogram, Scatter Plot, Line Plot
        - Bar Plot, Box Plot, Violin Plot
        - Heatmap, Pair Plot, Count Plot
        - Distribution Plot
        
        **Get started by uploading a CSV file in the sidebar!**
        """)

if __name__ == "__main__":
    main()