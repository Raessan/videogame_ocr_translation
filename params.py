# Parameters
# Region of the screen to capture. If it is None, the whole screen will be captured
region = (300, 780, 1340, 190)
# Threshold for binary image
threshold_binary = 160
# Threshold of zeros to consider new line
min_zeros_between_lines = 10
# Threshold of nonzeros in the vertical histogram to consider that they do not belong to a character (such as umlaut)
min_nonzeros_line = 25
# Threshold of zeros to consider a different char
min_zeros_between_characters = 1
# Threshold of zeros to consider a space
min_zeros_space = 19
# Folder to save the ground truth ("dataset")
folder_chars = "chars_german_cnn"
# Size of the images to compute HOG features
size_characters = (64, 64)
# Dictionary of chars to save the images using the alias (values of dictionary). If a char is not provided here, it is saved with the original name.
chars_dict = {
    'A': 'a_upper',
    'B': 'b_upper',
    'C': 'c_upper',
    'D': 'd_upper',
    'E': 'e_upper',
    'F': 'f_upper',
    'G': 'g_upper',
    'H': 'h_upper',
    'I': 'i_upper',
    'J': 'j_upper',
    'K': 'k_upper',
    'L': 'l_upper',
    'M': 'm_upper',
    'N': 'n_upper',
    'O': 'o_upper',
    'P': 'p_upper',
    'Q': 'q_upper',
    'R': 'r_upper',
    'S': 's_upper',
    'T': 't_upper',
    'U': 'u_upper',
    'V': 'v_upper',
    'W': 'w_upper',
    'X': 'x_upper',
    'Y': 'y_upper',
    'Z': 'z_upper',
    'ä': 'a_umlaut',
    'ö': 'o_umlaut',
    'ü': 'u_umlaut',
    'ß': 'eszett',
    'Ä': 'a_umlaut_upper',
    'Ö': 'o_umlaut_upper',
    'Ü': 'u_umlaut_upper',
    'ß': 'eszett',
    'é': 'e_acute',
    'à': 'a_grave',
    'è': 'e_grave',
    'â': 'a_circumflex',
    'ê': 'e_circumflex',
    'ô': 'o_circumflex',
    'û': 'u_circumflex',
    'î': 'i_circumflex',
    ',': 'comma',
    '.': 'period',
    ':': 'colon',
    ';': 'semicolon',
    '!': 'exclamation_mark',
    '?': 'question_mark',
    '-': 'hyphen',
    '_': 'underscore',
    "'": 'apostrophe',
    '"': 'quotation_mark',
    '(': 'left_parenthesis',
    ')': 'right_parenthesis',
    '[': 'left_square_bracket',
    ']': 'right_square_bracket',
    '{': 'left_curly_brace',
    '}': 'right_curly_brace',
    '<': 'less_than',
    '>': 'greater_than',
    '/': 'slash',
    '\\': 'backslash',
    '@': 'at_symbol',
    '#': 'hash',
    '$': 'dollar',
    '%': 'percent',
    '^': 'caret',
    '&': 'ampersand',
    '*': 'asterisk',
    '+': 'plus',
    '=': 'equals',
    ' ': 'cursor'
}
# Folder to save and load the NN
save_cnn_folder = "cnn_model"
save_cnn_file = "cnn_model.pth"
# File to save words and their translation
save_file_path = "words.txt"