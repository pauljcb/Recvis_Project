import os
from level_image_gen import LevelImageGen

from tokens import TOKEN_GROUPS, REPLACE_TOKENS

def load_level_from_text(path_to_level_txt, replace_tokens=REPLACE_TOKENS):
    """ Loads an ascii level from a text file. """
    with open(path_to_level_txt, "r") as f:
        ascii_level = []
        for line in f:
            for token, replacement in replace_tokens.items():
                line = line.replace(token, replacement)
            ascii_level.append(line)
    return ascii_level

if __name__ == '__main__':
    ImgGen = LevelImageGen('sprites/')

    for ii in range(50):
        lvl = load_level_from_text('generated_levels/'+str(ii)+'.txt')
        if lvl[-1][-1] == '\n':
            lvl[-1] = lvl[-1][0:-1]
        lvl_img = ImgGen.render(lvl)
        lvl_img.save('generated_levels/'+str(ii)+'_0.png', format='png')

