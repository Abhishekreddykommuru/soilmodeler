"""
Theme Configuration
Color schemes and styling constants for the application
"""

# Primary color palette
COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'success': '#4CAF50',
    'warning': '#FFC107',
    'danger': '#F44336',
    'info': '#4A90E2',
    'dark': '#262730',
    'light': '#F0F2F6',
    'white': '#FFFFFF'
}

# Gradient definitions
GRADIENTS = {
    'primary': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    'success': 'linear-gradient(135deg, #4CAF50 0%, #81C784 100%)',
    'warning': 'linear-gradient(135deg, #FFC107 0%, #FFD54F 100%)',
    'danger': 'linear-gradient(135deg, #F44336 0%, #E57373 100%)',
    'cool': 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
    'warm': 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)',
    'sunset': 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)'
}

# Font settings
FONTS = {
    'family': "'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
    'sizes': {
        'small': '0.875rem',
        'normal': '1rem',
        'large': '1.25rem',
        'xlarge': '1.5rem',
        'xxlarge': '2rem'
    }
}

# Spacing
SPACING = {
    'xs': '0.25rem',
    'sm': '0.5rem',
    'md': '1rem',
    'lg': '1.5rem',
    'xl': '2rem',
    'xxl': '3rem'
}

# Border radius
BORDER_RADIUS = {
    'small': '4px',
    'medium': '8px',
    'large': '12px',
    'xlarge': '16px',
    'round': '50%'
}

# Shadows
SHADOWS = {
    'small': '0 2px 4px rgba(0, 0, 0, 0.1)',
    'medium': '0 4px 6px rgba(0, 0, 0, 0.1)',
    'large': '0 10px 25px rgba(0, 0, 0, 0.15)',
    'xlarge': '0 20px 40px rgba(0, 0, 0, 0.2)'
}

# Component-specific styling
CARD_STYLE = f"""
    background: {COLORS['white']};
    border-radius: {BORDER_RADIUS['large']};
    padding: {SPACING['lg']};
    box-shadow: {SHADOWS['medium']};
    transition: all 0.3s ease;
"""

BUTTON_STYLE = f"""
    font-weight: 600;
    border-radius: {BORDER_RADIUS['medium']};
    padding: {SPACING['sm']} {SPACING['lg']};
    transition: all 0.3s ease;
    border: none;
    cursor: pointer;
"""

# Metric card colors based on value
def get_metric_color(value, metric_type='r2'):
    """
    Get color based on metric value
    
    Args:
        value: Metric value
        metric_type: Type of metric ('r2', 'rpd', etc.)
    
    Returns:
        Color hex code
    """
    if metric_type == 'r2':
        if value >= 0.85:
            return COLORS['success']
        elif value >= 0.75:
            return COLORS['warning']
        else:
            return COLORS['danger']
    elif metric_type == 'rpd':
        if value >= 2.5:
            return COLORS['success']
        elif value >= 2.0:
            return COLORS['warning']
        else:
            return COLORS['danger']
    else:
        return COLORS['info']

# Apply custom theme to Streamlit
def apply_custom_css():
    """
    Generate custom CSS for Streamlit application
    
    Returns:
        CSS string
    """
    return f"""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
        
        /* Global styles */
        html, body, [class*="css"] {{
            font-family: {FONTS['family']};
        }}
        
        /* Hide Streamlit branding */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {{
            width: 10px;
            height: 10px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: {COLORS['light']};
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: {COLORS['primary']};
            border-radius: {BORDER_RADIUS['small']};
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: {COLORS['secondary']};
        }}
        
        /* Metric cards */
        [data-testid="stMetricValue"] {{
            font-size: {FONTS['sizes']['xxlarge']};
            font-weight: 800;
            background: {GRADIENTS['primary']};
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        [data-testid="stMetricDelta"] {{
            font-weight: 600;
        }}
        
        /* Buttons */
        .stButton>button {{
            {BUTTON_STYLE}
        }}
        
        .stButton>button:hover {{
            transform: translateY(-2px);
            box-shadow: {SHADOWS['large']};
        }}
        
        .stButton>button[kind="primary"] {{
            background: {GRADIENTS['primary']};
            color: white;
        }}
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: {SPACING['lg']};
            background-color: transparent;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            font-weight: 600;
            padding: {SPACING['md']} {SPACING['xl']};
            border-radius: {BORDER_RADIUS['medium']} {BORDER_RADIUS['medium']} 0 0;
        }}
        
        .stTabs [data-baseweb="tab"][aria-selected="true"] {{
            background: {GRADIENTS['primary']};
            color: white;
        }}
        
        /* Dataframes */
        .dataframe {{
            border-radius: {BORDER_RADIUS['medium']};
            overflow: hidden;
            box-shadow: {SHADOWS['small']};
        }}
        
        /* Expanders */
        .streamlit-expanderHeader {{
            font-weight: 600;
            font-size: {FONTS['sizes']['large']};
            border-radius: {BORDER_RADIUS['medium']};
            background: {COLORS['light']};
        }}
        
        .streamlit-expanderHeader:hover {{
            background: {COLORS['primary']}20;
        }}
        
        /* Sidebar */
        .css-1d391kg {{
            background: {GRADIENTS['primary']};
        }}
        
        /* File uploader */
        [data-testid="stFileUploader"] {{
            border: 2px dashed {COLORS['primary']};
            border-radius: {BORDER_RADIUS['large']};
            padding: {SPACING['xl']};
        }}
        
        /* Selectbox and multiselect */
        .stSelectbox, .stMultiSelect {{
            border-radius: {BORDER_RADIUS['medium']};
        }}
        
        /* Slider */
        .stSlider [data-baseweb="slider"] {{
            border-radius: {BORDER_RADIUS['medium']};
        }}
        
        /* Progress bar */
        .stProgress > div > div > div > div {{
            background: {GRADIENTS['primary']};
        }}
        
        /* Success/Warning/Error boxes */
        .stSuccess {{
            background-color: {COLORS['success']}20;
            border-left: 5px solid {COLORS['success']};
            border-radius: {BORDER_RADIUS['medium']};
        }}
        
        .stWarning {{
            background-color: {COLORS['warning']}20;
            border-left: 5px solid {COLORS['warning']};
            border-radius: {BORDER_RADIUS['medium']};
        }}
        
        .stError {{
            background-color: {COLORS['danger']}20;
            border-left: 5px solid {COLORS['danger']};
            border-radius: {BORDER_RADIUS['medium']};
        }}
        
        .stInfo {{
            background-color: {COLORS['info']}20;
            border-left: 5px solid {COLORS['info']};
            border-radius: {BORDER_RADIUS['medium']};
        }}
        
        /* Custom cards */
        .custom-card {{
            {CARD_STYLE}
        }}
        
        .custom-card:hover {{
            transform: translateY(-4px);
            box-shadow: {SHADOWS['large']};
        }}
    </style>
    """
