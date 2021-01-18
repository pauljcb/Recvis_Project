import os
import numpy as np

from PIL import Image, ImageOps, ImageEnhance
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

def nssd(patch, template):
    """ Returns normalized sum of squared difference coeff between 2 patches """
    ssd = np.sum(np.linalg.norm(patch-template, axis=2)**2)
    sq_denom = np.sum(np.linalg.norm(patch, axis=2)**2)*np.sum(np.linalg.norm(template, axis=2)**2)
    nssd = ssd / np.sqrt(sq_denom)
    return nssd

class ImageToLevel:
    """ Generates Super Mario Bros. ascii levels from PIL Image files """

    def __init__(self, sprite_path):
        """ sprite_path: path to the folder of sprite files, e.g. 'mario/sprites/' """

        # Load Graphics (assumes sprite_path points to "img" folder of Mario-AI-Framework or provided sprites folder
        mariosheet = Image.open(os.path.join(sprite_path, 'smallmariosheet.png'))
        enemysheet = Image.open(os.path.join(sprite_path, 'enemysheet.png'))
        itemsheet = Image.open(os.path.join(sprite_path, 'itemsheet.png'))
        mapsheet = Image.open(os.path.join(sprite_path, 'mapsheet.png'))

        # Cut out the actual sprites:
        sprite_dict = dict()
        # Mario Sheet
        sprite_dict['M'] = mariosheet.crop((4*16, 0, 5*16, 16))

        # Enemy Sheet
        enemy_names = ['r', 'k', 'g', 'y', 'wings', '*', 'plant']
        for i, e in enumerate(enemy_names):
            sprite_dict[e] = enemysheet.crop((0, i*2*16, 16, (i+1)*2*16))

        sprite_dict['E'] = enemysheet.crop((16, 2*2*16, 2*16, 3*2*16))  # Set generic enemy to second goomba sprite
        sprite_dict['plant'] = enemysheet.crop((16, (len(enemy_names)-1)*2*16, 2*16, len(enemy_names)*2*16))

        # Item Sheet
        sprite_dict['shroom'] = itemsheet.crop((0, 0, 16, 16))
        sprite_dict['flower'] = itemsheet.crop((16, 0, 2*16, 16))
        sprite_dict['flower2'] = itemsheet.crop((0, 16, 16, 2*16))
        sprite_dict['1up'] = itemsheet.crop((16, 16, 2*16, 2*16))

        # Map Sheet
        map_names = ['-', 'X', '#', 'B', 'b', 'b2', 'S', 'L',
                     '?', 'dump', '@', 'Q', 'dump', '!', 'D', 'o',
                     'o2', 'o3', '<', '>', '[', ']', 'bg_sl_l', 'bg_top',
                     'bg_sl_r', 'bg_m_l', 'bg_m', 'bg_m_r', 'bush_l', 'bush_m', 'bush_r', 'cloud_l',
                     'cloud_m', 'cloud_r', 'cloud_b_l', 'cloud_b_m', 'cloud_b_r', 'waves', 'water', 'F_top',
                     'F_b', 'F', 'bg_sky', '%', '%_l', '%_r', '%_m', '|',
                     '1', '2', 'C', 'U', 'T', 't', 'dump', 'dump']

        sheet_length = (7, 8)
        sprite_counter = 0
        for i in range(sheet_length[0]):
            for j in range(sheet_length[1]):
                sprite_dict[map_names[sprite_counter]] = mapsheet.crop((j*16, i*16, (j+1)*16, (i+1)*16))
                sprite_counter += 1

        sprite_dict['@'] = sprite_dict['?']
        sprite_dict['!'] = sprite_dict['Q']

        self.sprite_dict = sprite_dict


    def prepare_sprites(self, level_path='input/level_1-1.txt'):
        """ Helper to make correct sprites and sprite sizes to draw into the image."""

        self.sprite_dict['-'] = self.sprite_dict['bg_sky']

        for sprite_key in ['?', '@', 'Q', '!', 'C', 'U', 'L']:  # Block/Brick hidden items
            if sprite_key == 'L':
                i_key = '1up'
            elif sprite_key in ['?', '@', 'U']:
                i_key = 'shroom'
            else:
                i_key = 'o'

            mask = self.sprite_dict[i_key].getchannel(3)
            mask = ImageEnhance.Brightness(mask).enhance(0.7)
            self.sprite_dict[sprite_key] = Image.composite(self.sprite_dict[i_key], self.sprite_dict[sprite_key], mask=mask)

        for sprite_key in ['g', 'k']:
            bg = self.sprite_dict['bg_sky']
            fg = self.sprite_dict[sprite_key].crop((0, 16, 16, 32))
            self.sprite_dict[sprite_key] = Image.alpha_composite(bg, fg)

        for sprite_key in ['1', '2']:  # Hidden block
            if sprite_key == '1':
                i_key = '1up'
            else:
                i_key = 'o'

            mask1 = self.sprite_dict['D'].getchannel(3)
            mask1 = ImageEnhance.Brightness(mask1).enhance(0.5)
            tmp_sprite = Image.composite(self.sprite_dict['D'], self.sprite_dict[sprite_key], mask=mask1)
            mask = self.sprite_dict[i_key].getchannel(3)
            mask = ImageEnhance.Brightness(mask).enhance(0.7)
            self.sprite_dict[sprite_key] = Image.composite(self.sprite_dict[i_key], tmp_sprite, mask=mask)

        sprite_set = set()
        ascii_level = load_level_from_text(level_path)
        len_level = len(ascii_level[-1])
        height_level = len(ascii_level)
        for y in range(height_level):
            for x in range(len_level):
                sprite_set.add(ascii_level[y][x])
        self.sprite_dict = { key: self.sprite_dict[key] for key in sprite_set }

        for sprite_key in self.sprite_dict.keys():
             self.sprite_dict[sprite_key] = self.sprite_dict[sprite_key].resize((8,8), Image.ANTIALIAS)
            #  self.sprite_dict[sprite_key].show()


    def get_ascii(self, img_path, patch_size=8):
        lvl_img = Image.open(img_path)
        w, h = lvl_img.size[0]/8, lvl_img.size[1]/8
        w, h = int(w), int(h)
        c = len(self.sprite_dict)
        features = np.zeros((c,w,h))
        keys = np.array(list(self.sprite_dict.keys()))
        for ii in range(len(keys)):
            sprite_key = keys[ii]
            template = np.asarray(self.sprite_dict[sprite_key], dtype=float)[:,:,0:3]
            for x in range(w):
                for y in range(h):
                    l, t, r, b = (8*x,8*y,8*x+8,8*y+8)
                    nssd_min = 1e5
                    for i in [-2,-1,0,1,2]:
                        for j in [-1,0,1]:
                            if l+i >= 0 and t+j >= 0 and r+i < lvl_img.size[0] and b+j < lvl_img.size[1]:
                                patch = np.asarray(lvl_img.crop((l+i,t+j,r+i,b+j)), dtype=float)[:,:,0:3]
                                nssd_val = nssd(patch, template)
                                if nssd_val < nssd_min: nssd_min=nssd_val
                    if sprite_key in ['k', 'g']: nssd_min *= 2
                    if sprite_key in ['U', '@']: nssd_min *= 1.6
                    if sprite_key in ['1', 'C', '!']: nssd_min *= 1.5
                    if sprite_key in ['t']: nssd_min *= 0.9
                    if sprite_key in ['S', '#']: nssd_min *= 0.7
                    if sprite_key in ['X'] and y>13: nssd_min *= 0.3
                    if sprite_key in ['X'] and y<=13: nssd_min *= 2.0
                    features[ii,x,y] = nssd_min
        # for x in range(w):
        #     for y in range(h):
        #         sky_index = np.where(keys=='-')
        #         t_index = np.where(keys=='t')
        #         if features[sky_index,x,y]/features[t_index,x,y]>0.5 and features[sky_index,x,y]/features[t_index,x,y]<1:
        #             features[t_index,x,y] = 0.5 * features[t_index,x,y]
        return keys[np.argmin(features, axis=0)]


ImgLev = ImageToLevel('sprites/')
ImgLev.prepare_sprites(level_path='input/lvl_1-1.txt')

for ii in range(50):
    print(ii, ' .....')
    ascii_lvl = ImgLev.get_ascii(img_path = 'generated_levels/'+str(ii)+'.png')
    with open('generated_levels/'+str(ii)+'.txt', 'w') as filehandle:
        for row in ascii_lvl.T:
            for col in row:
                filehandle.write(col)
            filehandle.write('\n')

