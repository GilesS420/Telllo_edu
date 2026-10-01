import keyboard

# Dictionary to store key states
key_states = {}

def init():
    """Initialize the keyboard listener"""
    pass

def getKey(key_name):
    """
    Check if a key is pressed
    
    Args:
        key_name: Name of the key (e.g., 'z', 'space', 'UP', 'DOWN')
    
    Returns:
        True if key is pressed, False otherwise
    """
    # Handle special key names
    key_map = {
        'space': 'space',
        'UP': 'up',
        'DOWN': 'down',
        'LEFT': 'left',
        'RIGHT': 'right',
        'esc': 'esc'
    }
    
    # Get the actual key name
    actual_key = key_map.get(key_name, key_name.lower())
    
    try:
        return keyboard.is_pressed(actual_key)
    except:
        return False
