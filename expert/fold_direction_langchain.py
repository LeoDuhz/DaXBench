from utils.Vector2D import *
import random

class S_Corner_Lefttop_Middle():
    @staticmethod
    def steps():
        return 1
    
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        return (pt_center(top_left, bottom_left), pt_center(top_left, top_right))
    
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        return [(top_left, pt_center(top_left, bottom_right)), (top_left, pt_center(top_left, bottom_right))]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the left top corner of the square into the center",
        "Fold the upper-left corner of the square towards the center.",
        "Take the top-left corner of the square and fold it inwards, towards the center.",
        "Bring the corner situated at the top-left of the square towards the center by folding.",
        "Position the square so that its top-left corner aligns with the center, folding it inward."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class S_Corner_Righttop_Middle():
    @staticmethod
    def steps():
        return 1
    
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        return (pt_center(top_right, top_left), pt_center(top_right, bottom_right))
    
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        return [(top_right, pt_center(top_right, bottom_left)), (top_right, pt_center(top_right, bottom_left))]
    
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the right top corner of the square into the center",
        "Fold the upper-right corner of the square towards the center.",
        "Take the top-right corner of the square and fold it inwards, towards the center.",
        "Bring the corner situated at the top-right of the square towards the center by folding.",
        "Position the square so that its top-right corner aligns with the center, folding it inward."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class S_Corner_Leftbottom_Middle():
    @staticmethod
    def steps():
        return 1
    
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        return (pt_center(bottom_left, bottom_right), pt_center(bottom_left, top_left))
    
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        return [(bottom_left, pt_center(bottom_left, top_right)), (bottom_left, pt_center(bottom_left, top_right))]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the left bottom corner of the square into the center",
        "Fold the bottom-left corner of the square towards the center.",
        "Take the bottom-left corner of the square and fold it inwards, towards the center.",
        "Bring the corner situated at the bottom-left of the square towards the center by folding.",
        "Position the square so that its bottom-left corner aligns with the center, folding it inward."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class S_Corner_Rightbottom_Middle():
    @staticmethod
    def steps():
        return 1
    
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        return (pt_center(bottom_right, top_right), pt_center(bottom_right, bottom_left))
    
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        return [(bottom_right, pt_center(bottom_right, top_left)), (bottom_right, pt_center(bottom_right, top_left))]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the right bottom corner of the square into the center",
        "Fold the bottom-right corner of the square towards the center.",
        "Take the bottom-right corner of the square and fold it inwards, towards the center.",
        "Bring the corner situated at the bottom-right of the square towards the center by folding.",
        "Position the square so that its bottom-right corner aligns with the center, folding it inward."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]
        
class S_Corner_Lefttop_Righttop_Middle():
    @staticmethod
    def steps():
        return 2
    
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return (pt_center(top_left, bottom_left), pt_center(top_left, top_right))
        else:
            return (pt_center(top_left, top_right), pt_center(top_right, bottom_right))
    
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return [(top_left, pt_center(top_left, bottom_right)), (top_left, pt_center(top_left, bottom_right))]
        else:
            return [(top_right, pt_center(top_right, bottom_left)), (top_right, pt_center(top_right, bottom_left))]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the left top and right top corners of the square into the center.",
        "Fold both the top-left and top-right corners of the square towards the center.",
        "Bring the corners at the top-left and top-right of the square towards the center and fold them inward.",
        "Fold the upper corners of the square, both left and right, towards the center point.",
        "Position the square so that both the top-left and top-right corners align with the center, folding them inward."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class S_Corner_Lefttop_Rightbottom_Middle():
    @staticmethod
    def steps():
        return 2
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return (pt_center(top_left, bottom_left), pt_center(top_left, top_right))
        else:
            return (pt_center(top_right, bottom_right), pt_center(bottom_left, bottom_right))
    
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return [(top_left, pt_center(top_left, bottom_right)), (top_left, pt_center(top_left, bottom_right))]
        else:
            return [(bottom_right, pt_center(top_left, bottom_right)), (bottom_right, pt_center(top_left, bottom_right))]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the left top and right bottom corners of the square into the center.",
        "Fold both the top-left and bottom-right corners of the square towards the center.",
        "Bring the corners at the top-left and bottom-right of the square towards the center and fold them inward.",
        "Fold inward the corners positioned at the top-left and bottom-right of the square, bringing them towards the center.",
        "Position the square so that both the top-left and bottom-right corners align with the center, folding them inward."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class S_Corner_Lefttop_Leftbottom_Middle():
    @staticmethod
    def steps():
        return 2
    
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return (pt_center(top_left, bottom_left), pt_center(top_left, top_right))
        else:
            return (pt_center(bottom_left, bottom_right), pt_center(top_left, bottom_left))
        
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return [(top_left, pt_center(top_left, bottom_right)), (top_left, pt_center(top_left, bottom_right))]
        else:
            return [(bottom_left, pt_center(bottom_left, top_right)), (bottom_left, pt_center(bottom_left, top_right))]
    
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the left top and left bottom corners of the square into the center.",
        "Fold both the top-left and bottom-left corners of the square towards the center.",
        "Bring the corners at the top-left and bottom-left of the square towards the center and fold them inward.",
        "Fold inward the corners positioned at the top-left and bottom-left of the square, bringing them towards the center.",
        "Position the square so that both the top-left and bottom-left corners align with the center, folding them inward."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class S_Corner_Righttop_Rightbottom_Middle():
    @staticmethod
    def steps():
        return 2
    
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return (pt_center(top_right, top_left), pt_center(top_right, bottom_right))
        else:
            return (pt_center(top_right, bottom_right), pt_center(bottom_right, bottom_left))
    
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return [(top_right, pt_center(top_right, bottom_left)), (top_right, pt_center(top_right, bottom_left))]
        else:
            return [(bottom_right, pt_center(bottom_right, top_left)), (bottom_right, pt_center(bottom_right, top_left))]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the right top and right bottom corners of the square into the center.",
        "Fold both the top-right and bottom-right corners of the square towards the center.",
        "Bring the corners at the top-right and bottom-right of the square towards the center and fold them inward.",
        "Fold inward the corners positioned at the top-right and bottom-right of the square, bringing them towards the center.",
        "Position the square so that both the top-right and bottom-right corners align with the center, folding them inward."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class S_Corner_Righttop_Leftbottom_Middle():
    @staticmethod
    def steps():
        return 2
    
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return (pt_center(top_right, top_left), pt_center(top_right, bottom_right))
        else:
            return (pt_center(bottom_left, bottom_right), pt_center(top_left, bottom_left))
    
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return [(top_right, pt_center(top_right, bottom_left)), (top_right, pt_center(top_right, bottom_left))]
        else:
            return [(bottom_left, pt_center(bottom_left, top_right)), (bottom_left, pt_center(bottom_left, top_right))]
    
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the right top and left bottom corners of the square into the center.",
        "Fold both the top-right and bottom-left corners of the square towards the center.",
        "Bring the corners at the top-right and bottom-left of the square towards the center and fold them inward.",
        "Fold inward the corners positioned at the top-right and bottom-left of the square, bringing them towards the center.",
        "Position the square so that both the top-right and bottom-left corners align with the center, folding them inward."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class S_Corner_Rightbottom_Leftbottom_Middle():
    @staticmethod
    def steps():
        return 2
    
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return (pt_center(bottom_right, top_right), pt_center(bottom_right, bottom_left))
        else:
            return (pt_center(bottom_left, bottom_right), pt_center(bottom_left, top_left))
        
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return [(bottom_right, pt_center(bottom_right, top_left)), (bottom_right, pt_center(bottom_right, top_left))]
        else:
            return [(bottom_left, pt_center(bottom_left, top_right)), (bottom_left, pt_center(bottom_left, top_right))]
    
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the right bottom and left bottom corners of the square into the center.",
        "Fold both the bottom-right and bottom-left corners of the square towards the center.",
        "Bring the corners at the bottom-right and bottom-left of the square towards the center and fold them inward.",
        "Fold inward the corners positioned at the bottom-right and bottom-left of the square, bringing them towards the center.",
        "Position the square so that both the bottom-right and bottom-left corners align with the center, folding them inward."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class S_Corner_All_Middle():
    @staticmethod
    def steps():
        return 4
    
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return (pt_center(top_left, bottom_left), pt_center(top_left, top_right))
        elif step == 1:
            return (pt_center(top_left, top_right), pt_center(top_right, bottom_right))
        elif step == 2:
            return (pt_center(top_right, bottom_right), pt_center(bottom_right, bottom_left))
        else:
            return (pt_center(bottom_right, bottom_left), pt_center(bottom_left, top_left))
    
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return [(top_left, pt_center(top_left, bottom_right)), (top_left, pt_center(top_left, bottom_right))]
        elif step == 1:
            return [(top_right, pt_center(top_right, bottom_left)), (top_right, pt_center(top_right, bottom_left))]
        elif step == 2:
            return [(bottom_right, pt_center(bottom_right, top_left)), (bottom_right, pt_center(bottom_right, top_left))]
        else:
            return [(bottom_left, pt_center(bottom_left, top_right)), (bottom_left, pt_center(bottom_left, top_right))]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold all corners of the square into the center.",
        "Fold all corners of the square towards the center.",
        "Bring all corners of the square towards the center.",
        "Fold inward all corners of the square, bringing them towards the center.",
        "Position the square so that all corners align with the center, folding them inward in 4 steps."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class S_Corner_All_Middle_Plot():
    @staticmethod
    def steps():
        return 4
    
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return (pt_center(top_left, bottom_left), pt_center(top_left, top_right))
        elif step == 1:
            return (pt_center(top_left, top_right), pt_center(top_right, bottom_right))
        elif step == 2:
            return (pt_center(top_right, bottom_right), pt_center(bottom_right, bottom_left))
        else:
            return (pt_center(bottom_right, bottom_left), pt_center(bottom_left, top_left))
    
    def modify_end_pos(self, start_pos, end_pos):
        end_pos = pt_sum(end_pos, pt_scale(line_vector(make_ln_from_pts(start_pos, end_pos)), -25))
        
        return (start_pos, end_pos)
    
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return [self.modify_end_pos(top_left, pt_center(top_left, bottom_right)), self.modify_end_pos(top_left, pt_center(top_left, bottom_right))]
        elif step == 1:
            return [self.modify_end_pos(top_right, pt_center(top_right, bottom_left)), self.modify_end_pos(top_right, pt_center(top_right, bottom_left))]
        elif step == 2:
            return [self.modify_end_pos(bottom_right, pt_center(bottom_right, top_left)), self.modify_end_pos(bottom_right, pt_center(bottom_right, top_left))]
        else:
            return [self.modify_end_pos(bottom_left, pt_center(bottom_left, top_right)), self.modify_end_pos(bottom_left, pt_center(bottom_left, top_right))]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold all corners of the square into the center.",
        "Fold all corners of the square towards the center.",
        "Bring all corners of the square towards the center and fold them inward.",
        "Fold inward all corners of the square, bringing them towards the center.",
        "Position the square so that all corners align with the center, folding them inward in 4 steps."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class S_Triangle_Lefttop_Rightbottom():
    @staticmethod
    def steps():
        return 1
    
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        return (bottom_left, top_right)
    
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        return [(top_left, bottom_right), (top_left, bottom_right)]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the left top corner of the square to the right bottom corner.",
        "Fold the top-left corner of the square to the bottom-right corner.",
        "Take the top-left corner of the square and fold it to the bottom-right corner.",
        "Bring the corner situated at the top-left of the square to the corner at the bottom-right by folding.",
        "Position the square so that its top-left corner aligns with the bottom-right corner."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class S_Triangle_Righttop_Leftbottom():
    @staticmethod
    def steps():
        return 1
    
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        return (top_left, bottom_right)
    
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        return [(top_right, bottom_left), (top_right, bottom_left)]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the right top corner of the square to the left bottom corner.",
        "Fold the top-right corner of the square to the bottom-left corner.",
        "Take the top-right corner of the square and fold it to the bottom-left corner.",
        "Bring the corner situated at the top-right of the square to the corner at the bottom-left by folding.",
        "Position the square so that its top-right corner aligns with the bottom-left corner."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class S_Triangle_Rightbottom_Lefttop():
    @staticmethod
    def steps():
        return 1
    
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        return (top_right, bottom_left)
    
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        return [(bottom_right, top_left), (bottom_right, top_left)]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the right bottom corner of the square to the left top corner.",
        "Fold the bottom-right corner of the square to the top-left corner.",
        "Take the bottom-right corner of the square and fold it to the top-left corner.",
        "Bring the corner situated at the bottom-right of the square to the corner at the top-left by folding.",
        "Position the square so that its bottom-right corner aligns with the top-left corner."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class S_Triangle_Leftbottom_Righttop():
    @staticmethod
    def steps():
        return 1
    
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        return (bottom_right, top_left)
    
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        return [(bottom_left, top_right), (bottom_left, top_right)]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the left bottom corner of the square to the right top corner.",
        "Fold the bottom-left corner of the square to the top-right corner.",
        "Take the bottom-left corner of the square and fold it to the top-right corner.",
        "Bring the corner situated at the bottom-left of the square to the corner at the top-right by folding.",
        "Position the square so that its bottom-left corner aligns with the top-right corner."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class S_Triangle_Lefttop_Rightbottom_Leftbottom_Righttop():
    @staticmethod
    def steps():
        return 2
    
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return (bottom_left, top_right)
        elif step == 1:
            return (bottom_right, pt_center(bottom_left, top_right))
        
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return [(top_left, bottom_right), (top_left, bottom_right)]
        elif step == 1:
            return [(bottom_left, top_right), (bottom_left, top_right)]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the top-left corner towards the bottom-right corner, and fold the bottom-left corner towards the top-right corner.",
        "Bring the top-left corner down to meet the bottom-right corner, then fold the bottom-left corner up to meet the top-right corner.",
        "Fold the top-left corner diagonally to meet the bottom-right corner, and similarly, fold the bottom-left corner diagonally to meet the top-right corner.",
        "Position the square such that the top-left corner folds over to the bottom-right corner, and the bottom-left corner folds over to the top-right corner.",
        "Converge the top-left corner towards the bottom-right corner, then bring the bottom-left corner upwards to meet the top-right corner."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class S_Triangle_Lefttop_Rightbottom_Righttop_Leftbottom():
    @staticmethod
    def steps():
        return 2
    
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return (bottom_left, top_right)
        elif step == 1:
            return (pt_center(bottom_left, top_right), bottom_right)
        
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return [(top_left, bottom_right), (top_left, bottom_right)]
        elif step == 1:
            return [(top_right, bottom_left), (top_right, bottom_left)]
    
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the top-left corner towards the bottom-right corner, and fold the top-right corner towards the bottom-left corner.",
        "Bring the top-left corner down to meet the bottom-right corner, then fold the top-right corner down to meet the bottom-left corner.",
        "Fold the top-left corner diagonally to meet the bottom-right corner, and similarly, fold the top-right corner diagonally to meet the bottom-left corner.",
        "Position the square such that the top-left corner folds over to the bottom-right corner, and the top-right corner folds over to the bottom-left corner.",
        "Converge the top-left corner towards the bottom-right corner, then bring the top-right corner downwards to meet the bottom-left corner."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]
        
class S_Triangle_Righttop_Leftbottom_Lefttop_Rightbottom():
    @staticmethod
    def steps():
        return 2
    
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return (top_left, bottom_right)
        elif step == 1:
            return (bottom_left, pt_center(top_left, bottom_right))
    
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return [(top_right, bottom_left), (top_right, bottom_left)]
        elif step == 1:
            return [(top_left, bottom_right), (top_left, bottom_right)]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the top-right corner towards the bottom-left corner, and fold the top-left corner towards the bottom-right corner.",
        "Bring the top-right corner down to meet the bottom-left corner, then fold the top-left corner down to meet the bottom-right corner.",
        "Fold the top-right corner diagonally to meet the bottom-left corner, and similarly, fold the top-left corner diagonally to meet the bottom-right corner.",
        "Position the square such that the top-right corner folds over to the bottom-left corner, and the top-left corner folds over to the bottom-right corner.",
        "Converge the top-right corner towards the bottom-left corner, then bring the top-left corner downwards to meet the bottom-right corner."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class S_Triangle_Righttop_Leftbottom_Rightbottom_Lefttop():
    @staticmethod
    def steps():
        return 2
    
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return (top_left, bottom_right)
        elif step == 1:
            return (pt_center(top_left, bottom_right), bottom_left)
        
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return [(top_right, bottom_left), (top_right, bottom_left)]
        elif step == 1:
            return [(bottom_right, top_left), (bottom_right, top_left)]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the top-right corner towards the bottom-left corner, and fold the bottom-right corner towards the top-left corner.",
        "Bring the top-right corner down to meet the bottom-left corner, then fold the bottom-right corner up to meet the top-left corner.",
        "Fold the top-right corner diagonally to meet the bottom-left corner, and similarly, fold the bottom-right corner diagonally to meet the top-left corner.",
        "Position the square such that the top-right corner folds over to the bottom-left corner, and the bottom-right corner folds over to the top-left corner.",
        "Converge the top-right corner towards the bottom-left corner, then bring the bottom-right corner upwards to meet the top-left corner."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class S_Triangle_Rightbottom_Lefttop_Leftbottom_Righttop():
    @staticmethod
    def steps():
        return 2
    
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return (top_right, bottom_left)
        elif step == 1:
            return (pt_center(top_right, bottom_left), top_left)
    
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return [(bottom_right, top_left), (bottom_right, top_left)]
        elif step == 1:
            return [(bottom_left, top_right), (bottom_left, top_right)]    
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the bottom-right corner towards the top-left corner, and fold the bottom-left corner towards the top-right corner.",
        "Fold the bottom-right corner diagonally to meet the top-left corner, and similarly, fold the bottom-left corner diagonally to meet the top-right corner.",
        "Position the square such that the bottom-right corner folds over to the top-left corner, and the bottom-left corner folds over to the top-right corner.",
        "Converge the bottom-right corner towards the top-left corner, then bring the bottom-left corner upwards to meet the top-right corner."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class S_Triangle_Rightbottom_Lefttop_Righttop_Leftbottom():
    @staticmethod
    def steps():
        return 2
    
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return (top_right, bottom_left)
        elif step == 1:
            return (top_left, pt_center(top_right, bottom_left))
        
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return [(bottom_right, top_left), (bottom_right, top_left)]
        elif step == 1:
            return [(top_right, bottom_left), (top_right, bottom_left)]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the bottom-right corner towards the top-left corner, and fold the top-right corner towards the bottom-left corner.",
        "Bring the bottom-right corner up to meet the top-left corner, then fold the top-right corner down to meet the bottom-left corner.",
        "Fold the bottom-right corner diagonally to meet the top-left corner, and similarly, fold the top-right corner diagonally to meet the bottom-left corner.",
        "Position the square such that the bottom-right corner folds over to the top-left corner, and the top-right corner folds over to the bottom-left corner.",
        "Converge the bottom-right corner towards the top-left corner, then bring the top-right corner downwards to meet the bottom-left corner."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class S_Triangle_Leftbottom_Righttop_Lefttop_Rightbottom():
    @staticmethod
    def steps():
        return 2
    
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return (bottom_right, top_left)
        elif step == 1:
            return (pt_center(top_left, bottom_right), top_right)
    
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return [(bottom_left, top_right), (bottom_left, top_right)]
        elif step == 1:
            return [(top_left, bottom_right), (top_left, bottom_right)]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the bottom-left corner towards the top-right corner, and fold the top-left corner towards the bottom-right corner.",
        "Bring the bottom-left corner up to meet the top-right corner, then fold the top-left corner down to meet the bottom-right corner.",
        "Fold the bottom-left corner diagonally to meet the top-right corner, and similarly, fold the top-left corner diagonally to meet the bottom-right corner.",
        "Position the square such that the bottom-left corner folds over to the top-right corner, and the top-left corner folds over to the bottom-right corner.",
        "Converge the bottom-left corner towards the top-right corner, then bring the top-left corner downwards to meet the bottom-right corner."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class S_Triangle_Leftbottom_Righttop_Rightbottom_Lefttop():
    @staticmethod
    def steps():
        return 2
    
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return (bottom_right, top_left)
        elif step == 1:
            return (pt_center(bottom_right, top_left), top_right)
    
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return [(bottom_left, top_right), (bottom_left, top_right)]
        elif step == 1:
            return [(bottom_right, top_left), (bottom_right, top_left)]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the bottom-left corner towards the top-right corner, and fold the bottom-right corner towards the top-left corner.",
        "Bring the bottom-left corner up to meet the top-right corner, then fold the bottom-right corner up to meet the top-left corner.",
        "Fold the bottom-left corner diagonally to meet the top-right corner, and similarly, fold the bottom-right corner diagonally to meet the top-left corner.",
        "Position the square such that the bottom-left corner folds over to the top-right corner, and the bottom-right corner folds over to the top-left corner.",
        "Converge the bottom-left corner towards the top-right corner, then bring the bottom-right corner upwards to meet the top-left corner."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]
        
class R_Edge_Top_Middle():
    @staticmethod
    def steps():
        return 1
    
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        return (pt_center(top_left, pt_center(top_left, bottom_left)), pt_center(top_right, pt_center(top_right, bottom_right)))
    
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        return [(top_left, pt_center(top_left, bottom_left)), (top_right, pt_center(top_right, bottom_right))]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the top-left-top-right line of the rectangle downwards to the horizontal middle-left-middle-right line of the rectangle.",
        "Fold the line from the top-left corner to the top-right corner downwards, aligning it with the horizontal line from the middle-left to the middle-right of the rectangle.",
        "Bring the line extending from the top-left corner to the top-right corner down to meet the horizontal line spanning from the middle-left to the middle-right of the rectangle.",
        "Fold the top-left to top-right line of the rectangle downwards, aligning it with the horizontal line drawn from the middle-left to the middle-right of the rectangle.",
        "Position the top-left to top-right line of the rectangle downwards, meeting it with the horizontal line stretching from the middle-left to the middle-right of the rectangle."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class R_Edge_Bottom_Middle():
    @staticmethod
    def steps():
        return 1
    
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        return (pt_center(bottom_right, pt_center(bottom_right, top_right)), pt_center(bottom_left, pt_center(bottom_left, top_left)))
    
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        return [(bottom_left, pt_center(bottom_left, top_left)), (bottom_right, pt_center(bottom_right, top_right))]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the bottom-left-bottom-right line of the rectangle upwards to the horizontal middle-left-middle-right line of the rectangle.",
        "Fold the line from the bottom-left corner to the bottom-right corner upwards, aligning it with the horizontal line from the middle-left to the middle-right of the rectangle.",
        "Bring the line extending from the bottom-left corner to the bottom-right corner up to meet the horizontal line spanning from the middle-left to the middle-right of the rectangle.",
        "Fold the bottom-left to bottom-right line of the rectangle upwards, aligning it with the horizontal line drawn from the middle-left to the middle-right of the rectangle.",
        "Position the bottom-left to bottom-right line of the rectangle upwards, meeting it with the horizontal line stretching from the middle-left to the middle-right of the rectangle."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class R_Edge_Left_Middle():
    @staticmethod
    def steps():
        return 1
    
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        return (pt_center(bottom_left, pt_center(bottom_left, bottom_right)), pt_center(top_left, pt_center(top_left, top_right)))
        
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        return [(bottom_left, pt_center(bottom_left, bottom_right)), (top_left, pt_center(top_left, top_right))]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the top-left-bottom-left line of the rectangle rightwards to the vertical middle-top-middle-bottom line of the rectangle.",
        "Fold the line from the top-left corner to the bottom-left corner rightwards, aligning it with the vertical line from the middle-top to the middle-bottom of the rectangle.",
        "Bring the line extending from the top-left corner to the bottom-left corner to the right to meet the vertical line spanning from the middle-top to the middle-bottom of the rectangle.",
        "Fold the top-left to bottom-left line of the rectangle rightwards, aligning it with the vertical line drawn from the middle-top to the middle-bottom of the rectangle.",
        "Position the top-left to bottom-left line of the rectangle rightwards, meeting it with the vertical line stretching from the middle-top to the middle-bottom of the rectangle."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]
        
class R_Edge_Right_Middle():
    @staticmethod
    def steps():
        return 1
    
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        return (pt_center(top_right, pt_center(top_right, top_left)), pt_center(bottom_right, pt_center(bottom_right, bottom_left)))
    
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        return [(top_right, pt_center(top_right, top_left)), (bottom_right, pt_center(bottom_right, bottom_left))]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the top-right-bottom-right line of the rectangle leftwards to the vertical middle-top-middle-bottom line of the rectangle.",
        "Fold the line from the top-right corner to the bottom-right corner leftwards, aligning it with the vertical line from the middle-top to the middle-bottom of the rectangle.",
        "Bring the line extending from the top-right corner to the bottom-right corner to the left to meet the vertical line spanning from the middle-top to the middle-bottom of the rectangle.",
        "Fold the top-right to bottom-right line of the rectangle leftwards, aligning it with the vertical line drawn from the middle-top to the middle-bottom of the rectangle.",
        "Position the top-right to bottom-right line of the rectangle leftwards, meeting it with the vertical line stretching from the middle-top to the middle-bottom of the rectangle."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class R_Edge_Top_Middle_Bottom_Middle():
    @staticmethod
    def steps():
        return 2
    
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return (pt_center(top_left, pt_center(top_left, bottom_left)), pt_center(top_right, pt_center(top_right, bottom_right)))
        elif step == 1:
            return (pt_center(bottom_right, pt_center(bottom_right, top_right)), pt_center(bottom_left, pt_center(bottom_left, top_left)))
    
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return [(top_left, pt_center(top_left, bottom_left)), (top_right, pt_center(top_right, bottom_right))]
        elif step == 1:
            return [(bottom_left, pt_center(bottom_left, top_left)), (bottom_right, pt_center(bottom_right, top_right))]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the top-left-top-right line of the rectangle downwards to the horizontal middle-left-middle-right line of the rectangle, and then fold the bottom-left-bottom-right line of the rectangle upwards to the horizontal middle-left-middle-right line of the rectangle.",
        # "Fold the line from the top-left corner to the top-right corner downwards, aligning it with the horizontal line from the middle-left to the middle-right of the rectangle, then fold the line from the bottom-left corner to the bottom-right corner upwards, aligning it with the horizontal line from the middle-left to the middle-right of the rectangle.",
        "Bring the top edge of the rectangle cloth downward to the horizontal middle line, and then repeat the same for the bottom edge.",
        # "Fold the top-left to top-right line of the rectangle downwards, aligning it with the horizontal line drawn from the middle-left to the middle-right of the rectangle, and then fold the bottom-left to bottom-right line of the rectangle upwards, aligning it with the horizontal line drawn from the middle-left to the middle-right of the rectangle.",
        # "Position the top-left to top-right line of the rectangle downwards, meeting it with the horizontal line stretching from the middle-left to the middle-right of the rectangle, and then position the bottom-left to bottom-right line of the rectangle upwards, meeting it with the horizontal line stretching from the middle-left to the middle-right of the rectangle."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class R_Edge_Bottom_Middle_Top_Middle():
    @staticmethod
    def steps():
        return 2
    
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return (pt_center(bottom_right, pt_center(bottom_right, top_right)), pt_center(bottom_left, pt_center(bottom_left, top_left)))
        elif step == 1:
            return (pt_center(top_left, pt_center(top_left, bottom_left)), pt_center(top_right, pt_center(top_right, bottom_right)))
                    
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return [(bottom_left, pt_center(bottom_left, top_left)), (bottom_right, pt_center(bottom_right, top_right))]
        elif step == 1:
            return [(top_left, pt_center(top_left, bottom_left)), (top_right, pt_center(top_right, bottom_right))]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the bottom-left-bottom-right line of the rectangle upwards to the horizontal middle-left-middle-right line of the rectangle, and then fold the top-left-top-right line of the rectangle downwards to the horizontal middle-left-middle-right line of the rectangle.",
        "Bring the bottom edge of the rectangle cloth upward to the horizontal middle line, and then repeat the same for the top edge.",
        # "Fold the line from the bottom-left corner to the bottom-right corner upwards, aligning it with the horizontal line from the middle-left to the middle-right of the rectangle, then fold the line from the top-left corner to the top-right corner downwards, aligning it with the horizontal line from the middle-left to the middle-right of the rectangle.",
        # "Bring the line extending from the bottom-left corner to the bottom-right corner up to meet the horizontal line spanning from the middle-left to the middle-right of the rectangle, and then bring the line extending from the top-left corner to the top-right corner down to meet the horizontal line spanning from the middle-left to the middle-right of the rectangle.",
        # "Fold the bottom-left to bottom-right line of the rectangle upwards, aligning it with the horizontal line drawn from the middle-left to the middle-right of the rectangle, and then fold the top-left to top-right line of the rectangle downwards, aligning it with the horizontal line drawn from the middle-left to the middle-right of the rectangle.",
        # "Position the bottom-left to bottom-right line of the rectangle upwards, meeting it with the horizontal line stretching from the middle-left to the middle-right of the rectangle, and then position the top-left to top-right line of the rectangle downwards, meeting it with the horizontal line stretching from the middle-left to the middle-right of the rectangle."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class R_Edge_Left_Middle_Right_Middle():
    @staticmethod
    def steps():
        return 2
        
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return (pt_center(bottom_left, pt_center(bottom_left, bottom_right)), pt_center(top_left, pt_center(top_left, top_right)))
        elif step == 1:
            return (pt_center(top_right, pt_center(top_right, top_left)), pt_center(bottom_right, pt_center(bottom_right, bottom_left)))
    
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return [(bottom_left, pt_center(bottom_left, bottom_right)), (top_left, pt_center(top_left, top_right))]
        elif step == 1:
            return [(top_right, pt_center(top_right, top_left)), (bottom_right, pt_center(bottom_right, bottom_left))]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the top-left-bottom-left line of the rectangle rightwards to the vertical middle-top-middle-bottom line of the rectangle, and then fold the top-right-bottom-right line of the rectangle leftwards to the vertical middle-top-middle-bottom line of the rectangle.",
        "Bring the left edge of the rectangular cloth rightwards to the vertical center line, and then repeat the same for the right edge.",
        # "Fold the line from the top-left corner to the bottom-left corner rightwards, aligning it with the vertical line from the middle-top to the middle-bottom of the rectangle, then fold the line from the top-right corner to the bottom-right corner leftwards, aligning it with the vertical line from the middle-top to the middle-bottom of the rectangle.",
        # "Bring the line extending from the top-left corner to the bottom-left corner to the right to meet the vertical line spanning from the middle-top to the middle-bottom of the rectangle, and then bring the line extending from the top-right corner to the bottom-right corner to the left to meet the vertical line spanning from the middle-top to the middle-bottom of the rectangle.",
        # "Fold the top-left to bottom-left line of the rectangle rightwards, aligning it with the vertical line drawn from the middle-top to the middle-bottom of the rectangle, and then fold the top-right to bottom-right line of the rectangle leftwards, aligning it with the vertical line drawn from the middle-top to the middle-bottom of the rectangle.",
        # "Position the top-left to bottom-left line of the rectangle rightwards, meeting it with the vertical line stretching from the middle-top to the middle-bottom of the rectangle, and then position the top-right to bottom-right line of the rectangle leftwards, meeting it with the vertical line stretching from the middle-top to the middle-bottom of the rectangle."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class R_Edge_Right_Middle_Left_Middle():
    @staticmethod
    def steps():
        return 2
        
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return (pt_center(top_right, pt_center(top_right, top_left)), pt_center(bottom_right, pt_center(bottom_right, bottom_left)))
        elif step == 1:
            return (pt_center(bottom_left, pt_center(bottom_left, bottom_right)), pt_center(top_left, pt_center(top_left, top_right)))
                    
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return [(top_right, pt_center(top_right, top_left)), (bottom_right, pt_center(bottom_right, bottom_left))]
        elif step == 1:
            return [(bottom_left, pt_center(bottom_left, bottom_right)), (top_left, pt_center(top_left, top_right))]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the top-right-bottom-right line of the rectangle leftwards to the vertical middle-top-middle-bottom line of the rectangle, and then fold the top-left-bottom-left line of the rectangle rightwards to the vertical middle-top-middle-bottom line of the rectangle.",
        # "Fold the line from the top-right corner to the bottom-right corner leftwards, aligning it with the vertical line from the middle-top to the middle-bottom of the rectangle, then fold the line from the top-left corner to the bottom-left corner rightwards, aligning it with the vertical line from the middle-top to the middle-bottom of the rectangle.",
        # "Bring the line extending from the top-right corner to the bottom-right corner to the left to meet the vertical line spanning from the middle-top to the middle-bottom of the rectangle, and then bring the line extending from the top-left corner to the bottom-left corner to the right to meet the vertical line spanning from the middle-top to the middle-bottom of the rectangle.",
        # "Fold the top-right to bottom-right line of the rectangle leftwards, aligning it with the vertical line drawn from the middle-top to the middle-bottom of the rectangle, and then fold the top-left to bottom-left line of the rectangle rightwards, aligning it with the vertical line drawn from the middle-top to the middle-bottom of the rectangle.",
        # "Position the top-right to bottom-right line of the rectangle leftwards, meeting it with the vertical line stretching from the middle-top to the middle-bottom of the rectangle, and then position the top-left to bottom-left line of the rectangle rightwards, meeting it with the vertical line stretching from the middle-top to the middle-bottom of the rectangle."
        "Position the right edge of the rectangular cloth leftwards, meeting it with the vertical center line, and then do the same for the left edge."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class R_Edge_Top_Bottom():
    @staticmethod
    def steps():
        return 1
        
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        return (pt_center(top_left, bottom_left), pt_center(top_right, bottom_right))
    
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        return [(top_left, bottom_left), (top_right, bottom_right)]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the top edge of the rectangle to the bottom edge.",
        "Bring the top side of the rectangle down to meet the bottom edge.",
        "Take the top edge of the rectangle and fold it to the bottom edge.",
        "Converge the top border of the rectangle towards the bottom edge by folding.",
        "Fold the top edge of the rectangle towards the bottom edge, aligning them."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class R_Edge_Bottom_Top():
    @staticmethod
    def steps():
        return 1
        
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        return (pt_center(bottom_right, top_right), pt_center(bottom_left, top_left))
    
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        return [(bottom_left, top_left), (bottom_right, top_right)]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the bottom edge of the rectangle to the top edge.",
        "Bring the bottom side of the rectangle up to meet the top edge.",
        "Take the bottom edge of the rectangle and fold it to the top edge.",
        "Converge the bottom border of the rectangle towards the top edge by folding.",
        "Fold the bottom edge of the rectangle towards the top edge, aligning them."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class R_Edge_Left_Right():
    @staticmethod
    def steps():
        return 1
        
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        return (pt_center(bottom_left, bottom_right), pt_center(top_left, top_right))
    
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        return [(bottom_left, bottom_right), (top_left, top_right)]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the left edge of the rectangle to the right edge.",
        "Bring the left side of the rectangle to meet the right edge.",
        "Take the left edge of the rectangle and fold it to the right edge.",
        "Converge the left border of the rectangle towards the right edge by folding.",
        "Fold the left edge of the rectangle towards the right edge, aligning them."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class R_Edge_Right_Left():
    @staticmethod
    def steps():
        return 1
        
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        return (pt_center(top_right, top_left), pt_center(bottom_right, bottom_left))
        
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        return [(top_right, top_left), (bottom_right, bottom_left)]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the right edge of the rectangle to the left edge.",
        "Bring the right side of the rectangle to meet the left edge.",
        "Take the right edge of the rectangle and fold it to the left edge.",
        "Converge the right border of the rectangle towards the left edge by folding.",
        "Fold the right edge of the rectangle towards the left edge, aligning them."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class R_Edge_Top_Bottom_Left_Right():
    @staticmethod
    def steps():
        return 2
    
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return (pt_center(top_left, bottom_left), pt_center(top_right, bottom_right))
        elif step == 1:
            return (pt_center(bottom_left, bottom_right), pt_center(top_left, top_right))
    
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return [(top_left, bottom_left), (top_right, bottom_right)]
        elif step == 1:
            return [(bottom_left, bottom_right), (pt_center(top_left, bottom_left), pt_center(top_right, bottom_right))]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the top edge of the rectangle to the bottom edge, then fold the left edge to the right edge.",
        "Bring the top side of the rectangle down to meet the bottom edge, then fold the left side to the right edge.",
        "Take the top edge of the rectangle and fold it towards the bottom edge, then fold the left edge towards the right edge.",
        "Fold the rectangle cloth in half from top to bottom, and then from left to right.",
        "Converge the top border of the rectangle towards the bottom edge by folding, then fold the left border to the right edge.",
        "Fold the top edge of the rectangle towards the bottom edge, aligning them, then fold the left edge towards the right edge."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class R_Edge_Top_Bottom_Right_Left():
    @staticmethod
    def steps():
        return 2
    
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return (pt_center(top_left, bottom_left), pt_center(top_right, bottom_right))
        elif step == 1:
            return (pt_center(top_right, top_left), pt_center(bottom_right, bottom_left))
            
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return [(top_left, bottom_left), (top_right, bottom_right)]
        elif step == 1:
            return [(bottom_right, bottom_left), (pt_center(top_right, bottom_right), pt_center(top_left, bottom_left))]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the top edge of the rectangle to the bottom edge, then fold the right edge to the left edge.",
        "Bring the top side of the rectangle down to meet the bottom edge, then fold the right side to the left edge.",
        "Take the top edge of the rectangle and fold it to the bottom edge, then fold the right edge to the left edge.",
        "Converge the top border of the rectangle towards the bottom edge by folding, then fold the right border to the left edge.",
        "Fold the top edge of the rectangle towards the bottom edge, aligning them, then fold the right edge towards the left edge."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class R_Edge_Bottom_Top_Left_Right():
    @staticmethod
    def steps():
        return 2
        
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return (pt_center(bottom_right, top_right), pt_center(bottom_left, top_left))
        elif step == 1:
            return (pt_center(bottom_left, bottom_right), pt_center(top_left, top_right))
    
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return [(bottom_left, top_left), (bottom_right, top_right)]
        elif step == 1:
            return [(top_left, top_right), (pt_center(bottom_left, top_left), pt_center(bottom_right, top_right))]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the bottom edge of the rectangle to the top edge, then fold the left edge to the right edge.",
        "Bring the bottom side of the rectangle up to meet the top edge, then fold the left side to the right edge.",
        "Take the bottom edge of the rectangle and fold it to the top edge, then fold the left edge to the right edge.",
        "Converge the bottom border of the rectangle towards the top edge by folding, then fold the left border to the right edge.",
        "Fold the bottom edge of the rectangle towards the top edge, aligning them, then fold the left edge towards the right edge."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class R_Edge_Bottom_Top_Right_Left():
    @staticmethod
    def steps():
        return 2
        
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return (pt_center(bottom_right, top_right), pt_center(bottom_left, top_left))
        elif step == 1:
            return (pt_center(top_right, top_left), pt_center(bottom_right, bottom_left))
    
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return [(bottom_right, top_right), (bottom_left, top_left)]
        elif step == 1:
            return [(top_right, top_left), (pt_center(bottom_right, top_right), pt_center(bottom_left, top_left))]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the bottom edge of the rectangle to the top edge, then fold the right edge to the left edge.",
        "Bring the bottom side of the rectangle up to meet the top edge, then fold the right side to the left edge.",
        "Take the bottom edge of the rectangle and fold it to the top edge, then fold the right edge to the left edge.",
        "Converge the bottom border of the rectangle towards the top edge by folding, then fold the right border to the left edge.",
        "Fold the bottom edge of the rectangle towards the top edge, aligning them, then fold the right edge towards the left edge."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]
        
class R_Edge_Left_Right_Top_Bottom():
    @staticmethod
    def steps():
        return 2
        
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return (pt_center(bottom_left, bottom_right), pt_center(top_left, top_right))
        elif step == 1:
            return (pt_center(top_left, bottom_left), pt_center(top_right, bottom_right))
    
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return [(bottom_left, bottom_right), (top_left, top_right)]
        elif step == 1:
            return [(pt_center(top_left, top_right), pt_center(bottom_left, bottom_right)), (top_right, bottom_right)]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the left edge of the rectangle to the right edge, then fold the top edge to the bottom edge.",
        "Bring the left side of the rectangle to meet the right edge, then fold the top side to the bottom edge.",
        "Take the left edge of the rectangle and fold it to the right edge, then fold the top edge to the bottom edge.",
        "Converge the left border of the rectangle towards the right edge by folding, then fold the top border to the bottom edge.",
        "Fold the left edge of the rectangle towards the right edge, aligning them, then fold the top edge towards the bottom edge."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class R_Edge_Left_Right_Bottom_Top():
    @staticmethod
    def steps():
        return 2
    
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return (pt_center(bottom_left, bottom_right), pt_center(top_left, top_right))
        elif step == 1:
            return (pt_center(bottom_right, top_right), pt_center(bottom_left, top_left))
    
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return [(bottom_left, bottom_right), (top_left, top_right)]
        elif step == 1:
            return [(bottom_right, top_right), (pt_center(bottom_left, bottom_right), pt_center(top_left, top_right))]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the left edge of the rectangle to the right edge, then fold the bottom edge to the top edge.",
        "Bring the left side of the rectangle to meet the right edge, then fold the bottom side to the top edge.",
        "Take the left edge of the rectangle and fold it to the right edge, then fold the bottom edge to the top edge.",
        "Converge the left border of the rectangle towards the right edge by folding, then fold the bottom border to the top edge.",
        "Fold the left edge of the rectangle towards the right edge, aligning them, then fold the bottom edge towards the top edge."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class R_Edge_Right_Left_Bottom_Top():
    @staticmethod
    def steps():
        return 2
        
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return (pt_center(top_right, top_left), pt_center(bottom_right, bottom_left))
        elif step == 1:
            return (pt_center(top_right, bottom_right), pt_center(top_left, bottom_left))
    
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return [(top_right, top_left), (bottom_right, bottom_left)]
        elif step == 1:
            return [(bottom_left, top_left), (pt_center(bottom_left, bottom_right), pt_center(top_left, top_right))]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the right edge of the rectangle to the left edge, then fold the bottom edge to the top edge.",
        "Bring the right side of the rectangle to meet the left edge, then fold the bottom side to the top edge.",
        "Take the right edge of the rectangle and fold it to the left edge, then fold the bottom edge to the top edge.",
        "Converge the right border of the rectangle towards the left edge by folding, then fold the bottom border to the top edge.",
        "Fold the right edge of the rectangle towards the left edge, aligning them, then fold the bottom edge towards the top edge."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class R_Edge_Right_Left_Top_Bottom():
    @staticmethod
    def steps():
        return 2
        
    def polyfold_symm_ln(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return (pt_center(top_right, top_left), pt_center(bottom_right, bottom_left))
        elif step == 1:
            return (pt_center(top_left, bottom_left), pt_center(top_right, bottom_right))
    
    def oracle_fold(self, top_left, top_right, bottom_right, bottom_left, step=0):
        if step == 0:
            return [(top_right, top_left), (bottom_right, bottom_left)]
        elif step == 1:
            return [(top_left, bottom_left), (pt_center(top_left, top_right), pt_center(bottom_left, bottom_right))]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the right edge of the rectangle to the left edge, then fold the top edge to the bottom edge.",
        "Bring the right side of the rectangle to meet the left edge, then fold the top side to the bottom edge.",
        "Take the right edge of the rectangle and fold it to the left edge, then fold the top edge to the bottom edge.",
        "Converge the right border of the rectangle towards the left edge by folding, then fold the top border to the bottom edge.",
        "Fold the right edge of the rectangle towards the left edge, aligning them, then fold the top edge towards the bottom edge."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class T_Sleeve_Left_Inwards():
    @staticmethod
    def steps():
        return 1
        
    def polyfold_symm_ln(self, bottom_left, left_armpit, left_sleeve_bottom, left_sleeve_top, left_shoulder_top, left_collar, spine_top, right_collar, right_shoulder_top, right_sleeve_top, right_sleeve_bottom, right_armpit, bottom_right, step=0):
        return (left_armpit, left_shoulder_top)
    
    def oracle_fold(self, bottom_left, left_armpit, left_sleeve_bottom, left_sleeve_top, left_shoulder_top, left_collar, spine_top, right_collar, right_shoulder_top, right_sleeve_top, right_sleeve_bottom, right_armpit, bottom_right, step=0):
        return [(left_sleeve_top, mirror_pt(left_sleeve_top, make_ln_from_pts(left_armpit, left_shoulder_top))), (left_sleeve_bottom, mirror_pt(left_sleeve_bottom, make_ln_from_pts(left_armpit, left_shoulder_top)))]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the left sleeve of the tshirt into the center with the left armpit-shoulder line as the symmetrical axis.",
        "Bring the left sleeve of the shirt inwards towards the center.",
        "Converge the left sleeve of the shirt towards the center by folding it inward.",
        "Position the shirt so that the left sleeve is folded inwardly, moving towards the center",
        "Fold the left sleeve inward, directing it towards the center of the shirt with the left armpit-shoulder line as the symmetrical axis."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class T_Sleeve_Right_Inwards():
    @staticmethod
    def steps():
        return 1
    
    def polyfold_symm_ln(self, bottom_left, left_armpit, left_sleeve_bottom, left_sleeve_top, left_shoulder_top, left_collar, spine_top, right_collar, right_shoulder_top, right_sleeve_top, right_sleeve_bottom, right_armpit, bottom_right, step=0):
        return (right_shoulder_top, right_armpit)
        
    def oracle_fold(self, bottom_left, left_armpit, left_sleeve_bottom, left_sleeve_top, left_shoulder_top, left_collar, spine_top, right_collar, right_shoulder_top, right_sleeve_top, right_sleeve_bottom, right_armpit, bottom_right, step=0):
        return [(right_sleeve_top, mirror_pt(right_sleeve_top, make_ln_from_pts(right_shoulder_top, right_armpit))), (right_sleeve_bottom, mirror_pt(right_sleeve_bottom, make_ln_from_pts(right_shoulder_top, right_armpit)))]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the right sleeve of the tshirt into the center with the right armpit-shoulder line as the symmetrical axis.",
        "Bring the right sleeve of the shirt inwards towards the center.",
        "Converge the right sleeve of the shirt towards the center by folding it inward.",
        "Position the shirt so that the right sleeve is folded inwardly, moving towards the center",
        "Fold the right sleeve inward, directing it towards the center of the shirt with the right armpit-shoulder line as the symmetrical axis."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]
        
class T_Sleeve_Left_Right_Inwards():
    @staticmethod
    def steps():
        return 2
    
    def polyfold_symm_ln(self, bottom_left, left_armpit, left_sleeve_bottom, left_sleeve_top, left_shoulder_top, left_collar, spine_top, right_collar, right_shoulder_top, right_sleeve_top, right_sleeve_bottom, right_armpit, bottom_right, step=0):
        if step == 0:
            return (left_armpit, left_shoulder_top)
        elif step == 1:
            return (right_shoulder_top, right_armpit)
            
    def oracle_fold(self, bottom_left, left_armpit, left_sleeve_bottom, left_sleeve_top, left_shoulder_top, left_collar, spine_top, right_collar, right_shoulder_top, right_sleeve_top, right_sleeve_bottom, right_armpit, bottom_right, step=0):
        if step == 0:
            return [(left_sleeve_top, mirror_pt(left_sleeve_top, make_ln_from_pts(left_armpit, left_shoulder_top))), (left_sleeve_bottom, mirror_pt(left_sleeve_bottom, make_ln_from_pts(left_armpit, left_shoulder_top)))]
        elif step == 1:
            return [(right_sleeve_top, mirror_pt(right_sleeve_top, make_ln_from_pts(right_shoulder_top, right_armpit))), (right_sleeve_bottom, mirror_pt(right_sleeve_bottom, make_ln_from_pts(right_shoulder_top, right_armpit)))]
    
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold left sleeve of the tshirt into the center with the armpit-shoulder lines as the symmetrical axes. Do the same for the right sleeve.",
        "Bring the left sleeve of the shirt inwards towards the center, and then bring the right sleeve inwards towards the center.",
        "Converge the left sleeve of the shirt towards the center by folding them inward. Do the same for the right sleeve",
        "Position the shirt so that first the left and then right sleeves are folded inwards, moving towards the center",
        "Fold first the left and then the right sleeves inward, directing them towards the center of the shirt with the armpit-shoulder lines as the symmetrical axes."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class T_Sleeve_Right_Left_Inwards():
    @staticmethod
    def steps():
        return 2
    
    def polyfold_symm_ln(self, bottom_left, left_armpit, left_sleeve_bottom, left_sleeve_top, left_shoulder_top, left_collar, spine_top, right_collar, right_shoulder_top, right_sleeve_top, right_sleeve_bottom, right_armpit, bottom_right, step=0):
        if step == 0:
            return (right_shoulder_top, right_armpit)
        elif step == 1:
            return (left_armpit, left_shoulder_top)
            
    def oracle_fold(self, bottom_left, left_armpit, left_sleeve_bottom, left_sleeve_top, left_shoulder_top, left_collar, spine_top, right_collar, right_shoulder_top, right_sleeve_top, right_sleeve_bottom, right_armpit, bottom_right, step=0):
        if step == 0:
            return [(right_sleeve_top, mirror_pt(right_sleeve_top, make_ln_from_pts(right_shoulder_top, right_armpit))), (right_sleeve_bottom, mirror_pt(right_sleeve_bottom, make_ln_from_pts(right_shoulder_top, right_armpit)))]
        elif step == 1:
            return [(left_sleeve_top, mirror_pt(left_sleeve_top, make_ln_from_pts(left_armpit, left_shoulder_top))), (left_sleeve_bottom, mirror_pt(left_sleeve_bottom, make_ln_from_pts(left_armpit, left_shoulder_top)))]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold right sleeve of the tshirt into the center with the armpit-shoulder lines as the symmetrical axes. Do the same for the left sleeve.",
        "Bring the right sleeve of the shirt inwards towards the center, and then bring the left sleeve inwards towards the center.",
        "Converge the right sleeve of the shirt towards the center by folding them inward. Do the same for the left sleeve",
        "Position the shirt so that first the right and then left sleeves are folded inwards, moving towards the center",
        "Fold first the right and then the left sleeves inward, directing them towards the center of the shirt with the armpit-shoulder lines as the symmetrical axes."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class T_Sleeve_Left_Half():
    @staticmethod
    def steps():
        return 1
        
    def polyfold_symm_ln(self, bottom_left, left_armpit, left_sleeve_bottom, left_sleeve_top, left_shoulder_top, left_collar, spine_top, right_collar, right_shoulder_top, right_sleeve_top, right_sleeve_bottom, right_armpit, bottom_right, step=0):
        return (pt_center(left_sleeve_bottom, left_armpit), pt_center(left_sleeve_top, left_shoulder_top))
        
    def oracle_fold(self, bottom_left, left_armpit, left_sleeve_bottom, left_sleeve_top, left_shoulder_top, left_collar, spine_top, right_collar, right_shoulder_top, right_sleeve_top, right_sleeve_bottom, right_armpit, bottom_right, step=0):
        return [(left_sleeve_top, left_shoulder_top), (left_sleeve_bottom, left_armpit)]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the left sleeve of the tshirt in half, aligning the left armpit-shoulder line with the left edge of the shirt.",
        "Bring the left sleeve of the shirt in half from left to right, aligning the left armpit-shoulder line with the left edge.",
        "Converge the left sleeve of the shirt in half, letting the left armpit-shoulder line meet the left edge.",
        "Position the shirt so that the left sleeve is folded in half, aligning the left armpit-shoulder line with the left edge",
        "Bend the left sleeve of the shirt in half, aligning the left armpit-shoulder line with the left edge of the shirt."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class T_Sleeve_Right_Half():
    @staticmethod
    def steps():
        return 1
        
    def polyfold_symm_ln(self, bottom_left, left_armpit, left_sleeve_bottom, left_sleeve_top, left_shoulder_top, left_collar, spine_top, right_collar, right_shoulder_top, right_sleeve_top, right_sleeve_bottom, right_armpit, bottom_right, step=0):
        return (pt_center(right_sleeve_top, right_shoulder_top), pt_center(right_sleeve_bottom, right_armpit))
        
    def oracle_fold(self, bottom_left, left_armpit, left_sleeve_bottom, left_sleeve_top, left_shoulder_top, left_collar, spine_top, right_collar, right_shoulder_top, right_sleeve_top, right_sleeve_bottom, right_armpit, bottom_right, step=0):
        return [(right_sleeve_top, right_shoulder_top), (right_sleeve_bottom, right_armpit)]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the right sleeve of the tshirt in half, aligning the right armpit-shoulder line with the right edge of the shirt.",
        "Bring the right sleeve of the shirt in half from right to left, aligning the right armpit-shoulder line with the right edge.",
        "Converge the right sleeve of the shirt in half, letting the right armpit-shoulder line meet the right edge.",
        "Position the shirt so that the right sleeve is folded in half, aligning the right armpit-shoulder line with the right edge",
        "Bend the right sleeve of the shirt in half, aligning the right armpit-shoulder line with the right edge of the shirt."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class T_Sleeve_Left_Half_Right_Half():
    @staticmethod
    def steps():
        return 2
        
    def polyfold_symm_ln(self, bottom_left, left_armpit, left_sleeve_bottom, left_sleeve_top, left_shoulder_top, left_collar, spine_top, right_collar, right_shoulder_top, right_sleeve_top, right_sleeve_bottom, right_armpit, bottom_right, step=0):
        if step == 0:
            return (pt_center(left_sleeve_bottom, left_armpit), pt_center(left_sleeve_top, left_shoulder_top))
        elif step == 1:
            return (pt_center(right_sleeve_top, right_shoulder_top), pt_center(right_sleeve_bottom, right_armpit))
    
    def oracle_fold(self, bottom_left, left_armpit, left_sleeve_bottom, left_sleeve_top, left_shoulder_top, left_collar, spine_top, right_collar, right_shoulder_top, right_sleeve_top, right_sleeve_bottom, right_armpit, bottom_right, step=0):
        if step == 0:
            return [(left_sleeve_top, left_shoulder_top), (left_sleeve_bottom, left_armpit)]
        elif step == 1:
            return [(right_sleeve_top, right_shoulder_top), (right_sleeve_bottom, right_armpit)]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the left sleeve of the tshirt in half, aligning the armpit-shoulder lines with the sleeve edges. Do the same for the right sleeve.",
        "Bring the left sleeve of the shirt in half, aligning the armpit-shoulder lines with the sleeve edges and then bring the right sleeve of the shirt in half, aligning the armpit-shoulder lines with the sleeve edges.",
        "Converge the left and right sleeves of t-shirt in half, letting the the sleeve edges meet the armpit-shoulder lines.",
        "Position the shirt so that the left sleeve is folded in half, aligning the armpit-shoulder lines with the sleeve edges. Repeat the same for the right sleeve.",
        "Bend first the left and then the right sleeves of the shirt in half, aligning the armpit-shoulder lines with the sleeve edges."
        ]       
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class T_Sleeve_Right_Half_Left_Half():
    @staticmethod
    def steps():
        return 2
        
    def polyfold_symm_ln(self, bottom_left, left_armpit, left_sleeve_bottom, left_sleeve_top, left_shoulder_top, left_collar, spine_top, right_collar, right_shoulder_top, right_sleeve_top, right_sleeve_bottom, right_armpit, bottom_right, step=0):
        if step == 0:
            return (pt_center(right_sleeve_top, right_shoulder_top), pt_center(right_sleeve_bottom, right_armpit))
        elif step == 1:
            return (pt_center(left_sleeve_bottom, left_armpit), pt_center(left_sleeve_top, left_shoulder_top))
    
    def oracle_fold(self, bottom_left, left_armpit, left_sleeve_bottom, left_sleeve_top, left_shoulder_top, left_collar, spine_top, right_collar, right_shoulder_top, right_sleeve_top, right_sleeve_bottom, right_armpit, bottom_right, step=0):
        if step == 0:
            return [(right_sleeve_top, right_shoulder_top), (right_sleeve_bottom, right_armpit)]
        elif step == 1:
            return [(left_sleeve_top, left_shoulder_top), (left_sleeve_bottom, left_armpit)]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the right sleeve of the tshirt in half, aligning the armpit-shoulder lines with the sleeve edges. Do the same for the left sleeve.",
        "Bring the right sleeve of the shirt in half, aligning the armpit-shoulder lines with the sleeve edges and then bring the left sleeve of the shirt in half, aligning the armpit-shoulder lines with the sleeve edges.",
        "Converge first the right and then the left sleeves of the shirt in half, letting the armpit-shoulder lines meet the sleeve edges.",
        "Position the shirt so that the right sleeve is folded in half, aligning the armpit-shoulder lines with the sleeve edges. Repeat the same for the left sleeve.",
        "Bend first the right and then the left sleeves of the shirt in half, aligning the armpit-shoulder lines with the sleeve edges."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class T_Sleeve_Left_Half_Inwards():
    @staticmethod
    def steps():
        return 2
        
    def polyfold_symm_ln(self, bottom_left, left_armpit, left_sleeve_bottom, left_sleeve_top, left_shoulder_top, left_collar, spine_top, right_collar, right_shoulder_top, right_sleeve_top, right_sleeve_bottom, right_armpit, bottom_right, step=0):
        if step == 0:
            return (pt_center(left_sleeve_bottom, left_armpit), pt_center(left_sleeve_top, left_shoulder_top))
        elif step == 1:
            return (left_armpit, left_shoulder_top)
    
    def oracle_fold(self, bottom_left, left_armpit, left_sleeve_bottom, left_sleeve_top, left_shoulder_top, left_collar, spine_top, right_collar, right_shoulder_top, right_sleeve_top, right_sleeve_bottom, right_armpit, bottom_right, step=0):
        if step == 0:
            return [(left_sleeve_top, left_shoulder_top), (left_sleeve_bottom, left_armpit)]
        elif step == 1:
            return [(pt_center(left_sleeve_top, left_shoulder_top), mirror_pt(pt_center(left_sleeve_top, left_shoulder_top), make_ln_from_pts(left_armpit, left_shoulder_top))), (pt_center(left_sleeve_bottom, left_armpit), mirror_pt(pt_center(left_sleeve_bottom, left_armpit), make_ln_from_pts(left_armpit, left_shoulder_top)))]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the left sleeve of the tshirt in half, aligning the left armpit-shoulder line with the left edge of the shirt, then fold the left sleeve inwards into the center.",
        "Bring the left sleeve of the shirt in half from left to right, aligning the left armpit-shoulder line with the left edge, then bend the left sleeve inwards towards the center.",
        "Converge the left sleeve of the shirt in half, letting the left armpit-shoulder line meet the left edge, then fold the left sleeve inwards towards the center.",
        "Position the shirt so that the left sleeve is folded in half, aligning the left armpit-shoulder line with the left edge, then bend the left sleeve inwards.",
        "Bend the left sleeve of the shirt in half, aligning the left armpit-shoulder line with the left edge of the shirt, then fold the left sleeve inwards towards the center."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class T_Sleeve_Right_Half_Inwards():
    @staticmethod
    def steps():
        return 2
        
    def polyfold_symm_ln(self, bottom_left, left_armpit, left_sleeve_bottom, left_sleeve_top, left_shoulder_top, left_collar, spine_top, right_collar, right_shoulder_top, right_sleeve_top, right_sleeve_bottom, right_armpit, bottom_right, step=0):
        if step == 0:
            return (pt_center(right_sleeve_top, right_shoulder_top), pt_center(right_sleeve_bottom, right_armpit))
        elif step == 1:
            return (right_shoulder_top, right_armpit)
    
    def oracle_fold(self, bottom_left, left_armpit, left_sleeve_bottom, left_sleeve_top, left_shoulder_top, left_collar, spine_top, right_collar, right_shoulder_top, right_sleeve_top, right_sleeve_bottom, right_armpit, bottom_right, step=0):
        if step == 0:
            return [(right_sleeve_top, right_shoulder_top), (right_sleeve_bottom, right_armpit)]
        elif step == 1:
            return [(pt_center(right_sleeve_top, right_shoulder_top), mirror_pt(pt_center(right_sleeve_top, right_shoulder_top), make_ln_from_pts(right_shoulder_top, right_armpit))), (pt_center(right_sleeve_bottom, right_armpit), mirror_pt(pt_center(right_sleeve_bottom, right_armpit), make_ln_from_pts(right_shoulder_top, right_armpit)))]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the right sleeve of the tshirt in half, aligning the right armpit-shoulder line with the right edge of the shirt, then fold the right sleeve inwards into the center.",
        "Bring the right sleeve of the shirt in half from right to left, aligning the right armpit-shoulder line with the right edge, then bend the right sleeve inwards towards the center.",
        "Converge the right sleeve of the shirt in half, letting the right armpit-shoulder line meet the right edge, then fold the right sleeve inwards towards the center.",
        "Position the shirt so that the right sleeve is folded in half, aligning the right armpit-shoulder line with the right edge, then bend the right sleeve inwards.",
        "Bend the right sleeve of the shirt in half, aligning the right armpit-shoulder line with the right edge of the shirt, then fold the right sleeve inwards towards the center."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]
        
class T_Sleeve_Left_Half_Inwards_Right_Half_Inwards():
    @staticmethod
    def steps():
        return 4
    
    def polyfold_symm_ln(self, bottom_left, left_armpit, left_sleeve_bottom, left_sleeve_top, left_shoulder_top, left_collar, spine_top, right_collar, right_shoulder_top, right_sleeve_top, right_sleeve_bottom, right_armpit, bottom_right, step=0):
        if step == 0:
            return (pt_center(left_sleeve_bottom, left_armpit), pt_center(left_sleeve_top, left_shoulder_top))
        elif step == 1:
            return (left_armpit, left_shoulder_top)
        elif step == 2:
            return (pt_center(right_sleeve_top, right_shoulder_top), pt_center(right_sleeve_bottom, right_armpit))
        elif step == 3:
            return (right_shoulder_top, right_armpit)
    
    def oracle_fold(self, bottom_left, left_armpit, left_sleeve_bottom, left_sleeve_top, left_shoulder_top, left_collar, spine_top, right_collar, right_shoulder_top, right_sleeve_top, right_sleeve_bottom, right_armpit, bottom_right, step=0):
        if step == 0:
            return [(left_sleeve_top, left_shoulder_top), (left_sleeve_bottom, left_armpit)]
        elif step == 1:
            return [(pt_center(left_sleeve_top, left_shoulder_top), mirror_pt(pt_center(left_sleeve_top, left_shoulder_top), make_ln_from_pts(left_armpit, left_shoulder_top))), (pt_center(left_sleeve_bottom, left_armpit), mirror_pt(pt_center(left_sleeve_bottom, left_armpit), make_ln_from_pts(left_armpit, left_shoulder_top)))]
        elif step == 2:
            return [(right_sleeve_top, right_shoulder_top), (right_sleeve_bottom, right_armpit)]
        elif step == 3:
            return [(pt_center(right_sleeve_top, right_shoulder_top), mirror_pt(pt_center(right_sleeve_top, right_shoulder_top), make_ln_from_pts(right_shoulder_top, right_armpit))), (pt_center(right_sleeve_bottom, right_armpit), mirror_pt(pt_center(right_sleeve_bottom, right_armpit), make_ln_from_pts(right_shoulder_top, right_armpit)))]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the left sleeve of the tshirt in half, aligning the left armpit-shoulder line with the left edge of the shirt, then fold the left sleeve inwards into the center. Repeat the same for the right sleeve.",
        "Bring the left sleeve of the shirt in half from left to right, aligning the left armpit-shoulder line with the left edge, then bend the left sleeve inwards towards the center. Repeat the same for the right sleeve.",
        "Converge the left sleeve of the shirt in half, letting the left armpit-shoulder line meet the left edge, then fold the left sleeve inwards towards the center. Repeat the same for the right sleeve.",
        "Position the shirt so that the left sleeve is folded in half, aligning the left armpit-shoulder line with the left edge, then bend the left sleeve inwards. Repeat the same for the right sleeve.",
        "Bend the left sleeve of the shirt in half, aligning the left armpit-shoulder line with the left edge of the shirt, then fold the left sleeve inwards towards the center. Repeat the same for the right sleeve."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class T_Sleeve_Right_Half_Inwards_Left_Half_Inwards():
    @staticmethod
    def steps():
        return 4
        
    def polyfold_symm_ln(self, bottom_left, left_armpit, left_sleeve_bottom, left_sleeve_top, left_shoulder_top, left_collar, spine_top, right_collar, right_shoulder_top, right_sleeve_top, right_sleeve_bottom, right_armpit, bottom_right, step=0):
        if step == 0:
            return (pt_center(right_sleeve_top, right_shoulder_top), pt_center(right_sleeve_bottom, right_armpit))
        elif step == 1:
            return (right_shoulder_top, right_armpit)
        elif step == 2:
            return (pt_center(left_sleeve_bottom, left_armpit), pt_center(left_sleeve_top, left_shoulder_top))
        elif step == 3:
            return (left_armpit, left_shoulder_top)
    
    def oracle_fold(self, bottom_left, left_armpit, left_sleeve_bottom, left_sleeve_top, left_shoulder_top, left_collar, spine_top, right_collar, right_shoulder_top, right_sleeve_top, right_sleeve_bottom, right_armpit, bottom_right, step=0):
        if step == 0:
            return [(right_sleeve_top, right_shoulder_top), (right_sleeve_bottom, right_armpit)]
        elif step == 1:
            return [(pt_center(right_sleeve_top, right_shoulder_top), mirror_pt(pt_center(right_sleeve_top, right_shoulder_top), make_ln_from_pts(right_shoulder_top, right_armpit))), (pt_center(right_sleeve_bottom, right_armpit), mirror_pt(pt_center(right_sleeve_bottom, right_armpit), make_ln_from_pts(right_shoulder_top, right_armpit)))]
        elif step == 2:
            return [(left_sleeve_top, left_shoulder_top), (left_sleeve_bottom, left_armpit)]
        elif step == 3:
            return [(pt_center(left_sleeve_top, left_shoulder_top), mirror_pt(pt_center(left_sleeve_top, left_shoulder_top), make_ln_from_pts(left_armpit, left_shoulder_top))), (pt_center(left_sleeve_bottom, left_armpit), mirror_pt(pt_center(left_sleeve_bottom, left_armpit), make_ln_from_pts(left_armpit, left_shoulder_top)))]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the right sleeve of the tshirt in half, aligning the right armpit-shoulder line with the right edge of the shirt, then fold the right sleeve inwards into the center. Repeat the same for the left sleeve.",
        "Bring the right sleeve of the shirt in half from right to left, aligning the right armpit-shoulder line with the right edge, then bend the right sleeve inwards towards the center. Repeat the same for the left sleeve.",
        "Converge the right sleeve of the shirt in half, letting the right armpit-shoulder line meet the right edge, then fold the right sleeve inwards towards the center. Repeat the same for the left sleeve.",
        "Position the shirt so that the right sleeve is folded in half, aligning the right armpit-shoulder line with the right edge, then bend the right sleeve inwards. Repeat the same for the left sleeve.",
        "Bend the right sleeve of the shirt in half, aligning the right armpit-shoulder line with the right edge of the shirt, then fold the right sleeve inwards towards the center. Repeat the same for the left sleeve."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]
        

class T_Half_Left_Right():
    @staticmethod
    def steps():
        return 1
        
    def polyfold_symm_ln(self, bottom_left, left_armpit, left_sleeve_bottom, left_sleeve_top, left_shoulder_top, left_collar, spine_top, right_collar, right_shoulder_top, right_sleeve_top, right_sleeve_bottom, right_armpit, bottom_right, step=0):
        return (pt_center(bottom_left, bottom_right), spine_top)
        
    def oracle_fold(self, bottom_left, left_armpit, left_sleeve_bottom, left_sleeve_top, left_shoulder_top, left_collar, spine_top, right_collar, right_shoulder_top, right_sleeve_top, right_sleeve_bottom, right_armpit, bottom_right, step=0):
        return [(pt_center(left_sleeve_bottom, left_sleeve_top), pt_center(right_sleeve_bottom, right_sleeve_top)), (bottom_left, bottom_right)]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the tshirt in half from left to right.",
        "Fold the t-shirt in half horizontally, from left to right.",
        "Bring the left side of the t-shirt over to meet the right side, folding it in half.",
        "Fold the t-shirt across its width, aligning the left edge with the right edge.",
        "Create a horizontal fold in the t-shirt, halving it from left to right."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class T_Half_Right_Left():
    @staticmethod
    def steps():
        return 1
        
    def polyfold_symm_ln(self, bottom_left, left_armpit, left_sleeve_bottom, left_sleeve_top, left_shoulder_top, left_collar, spine_top, right_collar, right_shoulder_top, right_sleeve_top, right_sleeve_bottom, right_armpit, bottom_right, step=0):
        return (spine_top, pt_center(bottom_left, bottom_right))
        
    def oracle_fold(self, bottom_left, left_armpit, left_sleeve_bottom, left_sleeve_top, left_shoulder_top, left_collar, spine_top, right_collar, right_shoulder_top, right_sleeve_top, right_sleeve_bottom, right_armpit, bottom_right, step=0):
        return [(pt_center(right_sleeve_bottom, right_sleeve_top), pt_center(left_sleeve_bottom, left_sleeve_top)), (bottom_right, bottom_left)]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the tshirt in half from right to left.",
        "Fold the t-shirt in half horizontally, from right to left.",
        "Bring the right side of the t-shirt over to meet the left side, folding it in half.",
        "Fold the t-shirt across its width, aligning the right edge with the left edge.",
        "Create a horizontal fold in the t-shirt, halving it from right to left."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class T_Half_Bottom_Top():
    @staticmethod
    def steps():
        return 1
        
    def polyfold_symm_ln(self, bottom_left, left_armpit, left_sleeve_bottom, left_sleeve_top, left_shoulder_top, left_collar, spine_top, right_collar, right_shoulder_top, right_sleeve_top, right_sleeve_bottom, right_armpit, bottom_right, step=0):
        return (pt_center(right_shoulder_top, bottom_right), pt_center(left_shoulder_top, bottom_left))
        
    def oracle_fold(self, bottom_left, left_armpit, left_sleeve_bottom, left_sleeve_top, left_shoulder_top, left_collar, spine_top, right_collar, right_shoulder_top, right_sleeve_top, right_sleeve_bottom, right_armpit, bottom_right, step=0):
        return [(bottom_left, left_shoulder_top), (bottom_right, right_shoulder_top)]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the bottom edge of the tshirt upwards to meet the top shoulder.",
        "Fold the bottom edge of the t-shirt upwards, bringing it to meet the top shoulder.",
        "Bring the bottom edge of the t-shirt upwards to meet the top shoulder, folding it.",
        "Fold the lower edge of the t-shirt upwards, aligning it with the top shoulder.",
        "Position the bottom edge of the t-shirt upwards, meeting it with the top shoulder as you fold."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class T_Block_Left_Right_Bottom_Top():
    @staticmethod
    def steps():
        return 3
        
    def polyfold_symm_ln(self, bottom_left, left_armpit, left_sleeve_bottom, left_sleeve_top, left_shoulder_top, left_collar, spine_top, right_collar, right_shoulder_top, right_sleeve_top, right_sleeve_bottom, right_armpit, bottom_right, step=0):
        if step == 0:
            return (left_armpit, left_shoulder_top)
        elif step == 1:
            return (right_shoulder_top, right_armpit)
        elif step == 2:
            return (pt_center(right_shoulder_top, bottom_right), pt_center(left_shoulder_top, bottom_left))
    
    def oracle_fold(self, bottom_left, left_armpit, left_sleeve_bottom, left_sleeve_top, left_shoulder_top, left_collar, spine_top, right_collar, right_shoulder_top, right_sleeve_top, right_sleeve_bottom, right_armpit, bottom_right, step=0):
        if step == 0:
            return [(left_sleeve_top, mirror_pt(left_sleeve_top, make_ln_from_pts(left_armpit, left_shoulder_top))), (left_sleeve_bottom, mirror_pt(left_sleeve_bottom, make_ln_from_pts(left_armpit, left_shoulder_top)))]
        elif step == 1:
            return [(right_sleeve_top, mirror_pt(right_sleeve_top, make_ln_from_pts(right_armpit, right_shoulder_top))), (right_sleeve_bottom, mirror_pt(right_sleeve_bottom, make_ln_from_pts(right_armpit, right_shoulder_top)))]
        elif step == 2:
            return [(bottom_left, left_shoulder_top), (bottom_right, right_shoulder_top)]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the left sleeve of the tshirt inwards into the center, then fold the right sleeve inwards into the center. Fold the bottom of the tshirt upwards to meet the top.",
        "Begin by folding the left sleeve of the t-shirt towards the center, followed by folding the right sleeve inward into the center. Then, fold the bottom of the t-shirt upwards to meet the top.",
        "First, fold the left sleeve of the t-shirt inward towards the center, then fold the right sleeve inward into the center. Finally, fold the bottom of the t-shirt upwards, bringing it to meet the top.",
        "Fold the left sleeve of the t-shirt towards the center, then fold the right sleeve towards the center. Lastly, fold the bottom of the t-shirt upwards to the top.",
        "Start by folding the left sleeve of the t-shirt into the center, followed by folding the right sleeve into the center. Finish by folding the bottom of the t-shirt upwards, meeting the top.",
        "Fold both sleeves of the t-shirt towards the center, followed by folding the t-shirt in half from bottom to top."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

# class T_Block_Right_Left_Top_Bottom():
#     def get_instruction(self, index=-1):
#         self.instructions = [
#         "Fold the right sleeve of the tshirt inwards into the center, then fold the left sleeve inwards into the center. Fold the top of the tshirt downwards to meet the bottom.",
#         "Begin by folding the right sleeve of the t-shirt inward towards the center, followed by folding the left sleeve inward into the center. Then, fold the top of the t-shirt downwards to meet the bottom.",
#         "First, fold the right sleeve of the t-shirt inward towards the center, then fold the left sleeve inward into the center. Finally, fold the top of the t-shirt downwards, bringing it to meet the bottom.",
#         "Fold the right sleeve of the t-shirt towards the center, then fold the left sleeve towards the center. Lastly, fold the top of the t-shirt downwards to the bottom.",
#         "Start by folding the right sleeve of the t-shirt into the center, followed by folding the left sleeve into the center. Finish by folding the top of the t-shirt downwards, meeting the bottom.",
#         ]
#         if index == -1:
#             return random.choice(self.instructions)
#         else:
#             return self.instructions[index%len(self.instructions)]

# class T_Block_Left_Right_Top_Bottom():
#     def get_instruction(self, index=-1):
#         self.instructions = [
#         "Fold the left sleeve of the tshirt inwards into the center, then fold the right sleeve inwards into the center. Fold the top of the tshirt downwards to meet the bottom.",
#         "Begin by folding the left sleeve of the t-shirt inward towards the center, followed by folding the right sleeve inward into the center. Then, fold the top of the t-shirt downwards to meet the bottom.",
#         "First, fold the left sleeve of the t-shirt inward towards the center, then fold the right sleeve inward into the center. Finally, fold the top of the t-shirt downwards, bringing it to meet the bottom.",
#         "Fold the left sleeve of the t-shirt towards the center, then fold the right sleeve towards the center. Lastly, fold the top of the t-shirt downwards to the bottom.",
#         "Start by folding the left sleeve of the t-shirt into the center, followed by folding the right sleeve into the center. Finish by folding the top of the t-shirt downwards, meeting the bottom.",
#         ]
#         if index == -1:
#             return random.choice(self.instructions)
#         else:
#             return self.instructions[index%len(self.instructions)]

class T_Block_Right_Left_Bottom_Top():
    @staticmethod
    def steps():
        return 3
        
    def polyfold_symm_ln(self, bottom_left, left_armpit, left_sleeve_bottom, left_sleeve_top, left_shoulder_top, left_collar, spine_top, right_collar, right_shoulder_top, right_sleeve_top, right_sleeve_bottom, right_armpit, bottom_right, step=0):
        if step == 0:
            return (right_shoulder_top, right_armpit)
        elif step == 1:
            return (left_armpit, left_shoulder_top)
        elif step == 2:
            return (pt_center(right_shoulder_top, bottom_right), pt_center(left_shoulder_top, bottom_left))
            
    def oracle_fold(self, bottom_left, left_armpit, left_sleeve_bottom, left_sleeve_top, left_shoulder_top, left_collar, spine_top, right_collar, right_shoulder_top, right_sleeve_top, right_sleeve_bottom, right_armpit, bottom_right, step=0):
        if step == 0:
            return [(right_sleeve_top, mirror_pt(right_sleeve_top, make_ln_from_pts(right_armpit, right_shoulder_top))), (right_sleeve_bottom, mirror_pt(right_sleeve_bottom, make_ln_from_pts(right_armpit, right_shoulder_top)))]
        elif step == 1:
            return [(left_sleeve_top, mirror_pt(left_sleeve_top, make_ln_from_pts(left_shoulder_top, left_armpit))), (left_sleeve_bottom, mirror_pt(left_sleeve_bottom, make_ln_from_pts(left_shoulder_top, left_armpit)))]
        elif step == 2:
            return [(bottom_right, right_shoulder_top), (bottom_left, left_shoulder_top)]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the right sleeve of the tshirt inwards into the center, then fold the left sleeve inwards into the center. Fold the bottom of the tshirt upwards to meet the top.",
        "First, fold the right sleeve of the T-shirt toward the center along the armpit-to-shoulder line, then do the same with the left sleeve. Finally, fold the T-shirt in half from bottom to top.",
        "Begin by folding the right sleeve of the t-shirt inward towards the center, followed by folding the left sleeve inward into the center. Then, fold the bottom of the t-shirt upwards to meet the top.",
        "First, fold the right sleeve of the t-shirt inward towards the center, then fold the left sleeve inward into the center. Finally, fold the bottom of the t-shirt upwards, bringing it to meet the top.",
        "Fold the right sleeve of the t-shirt towards the center, then fold the left sleeve towards the center. Lastly, fold the bottom of the t-shirt upwards to the top.",
        "Start by folding the right sleeve of the t-shirt into the center, followed by folding the left sleeve into the center. Finish by folding the bottom of the t-shirt upwards, meeting the top.",
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]
        
class P_Half_Left_Right():
    @staticmethod
    def steps():
        return 1
        
    def polyfold_symm_ln(self, left_leg_right, left_leg_left, top_left, top_right, right_leg_right, right_leg_left,crotch, step=0):
        return (pt_center(left_leg_left, right_leg_right), pt_center(top_left, top_right))
    
    def oracle_fold(self, left_leg_right, left_leg_left, top_left, top_right, right_leg_right, right_leg_left,crotch, step=0):
        return [(left_leg_left, right_leg_right), (top_left, top_right)]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the pant in half from left to right.",
        "Fold the pants in half horizontally, from left to right.",
        "Bring the left side of the pants over to meet the right side, folding them in half.",
        "Fold the pants across their width, aligning the left edge with the right edge.",
        "Create a horizontal fold in the pants, halving them from left to right."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class P_Half_Right_Left():
    @staticmethod
    def steps():
        return 1
    
    def polyfold_symm_ln(self, left_leg_right, left_leg_left, top_left, top_right, right_leg_right, right_leg_left,crotch, step=0):
        return (pt_center(top_right, top_left), pt_center(right_leg_right, left_leg_left))
    
    def oracle_fold(self, left_leg_right, left_leg_left, top_left, top_right, right_leg_right, right_leg_left,crotch, step=0):
        return [(right_leg_right, left_leg_left), (top_right, top_left)]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the pant in half from right to left.",
        "Fold the pants in half horizontally, from right to left.",
        "Bring the right side of the pants over to meet the left side, folding them in half.",
        "Fold the pants across their width, aligning the right edge with the left edge.",
        "Create a horizontal fold in the pants, halving them from right to left."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]
        
class P_Half_Leg_Left():
    @staticmethod
    def steps():
        return 1
    
    def polyfold_symm_ln(self, left_leg_right, left_leg_left, top_left, top_right, right_leg_right, right_leg_left,crotch, step=0):
        return (intercept(perpendicular(make_ln_from_pts(top_left, left_leg_left), pt_center(top_left, left_leg_left)), make_ln_from_pts(left_leg_right, crotch)), pt_center(top_left, left_leg_left))
        
    def oracle_fold(self, left_leg_right, left_leg_left, top_left, top_right, right_leg_right, right_leg_left,crotch, step=0):
        return [(left_leg_left, top_left), (left_leg_right, pt_center(top_left, top_right))]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the left part of the pant in half from the bottom left leg edge to the left top waist.",
        "Fold the left side of the pants in half from the bottom left leg edge up to the left top waist.",
        "Bring the bottom left leg edge of the pants up to the left top waist, folding the left side in half.",
        "Fold the left part of the pants in half, starting from the bottom left leg edge and extending up to the left top waist.",
        "Position the bottom left leg edge of the pants to meet the left top waist, folding the left part in half."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class P_Half_Leg_Right():
    @staticmethod
    def steps():
        return 1
        
    def polyfold_symm_ln(self, left_leg_right, left_leg_left, top_left, top_right, right_leg_right, right_leg_left,crotch, step=0):
        return (pt_center(top_right, right_leg_right), intercept(perpendicular(make_ln_from_pts(top_right, right_leg_right), pt_center(top_right, right_leg_right)), make_ln_from_pts(right_leg_left, crotch)))
    
    def oracle_fold(self, left_leg_right, left_leg_left, top_left, top_right, right_leg_right, right_leg_left,crotch, step=0):
        return [(right_leg_right, top_right), (right_leg_left, pt_center(top_left, top_right))]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the right part of the pant in half from the bottom right leg edge to the right top waist.",
        "Fold the right part of the pant in half from bottom to top by folding the bottom right leg right corner to the right top waist.",
        "Fold the right side of the pants in half from the bottom right leg edge up to the right top waist.",
        "Bring the bottom right leg edge of the pants up to the right top waist, folding the right side in half.",
        "Fold the right part of the pants in half, starting from the bottom right leg edge and extending up to the right top waist.",
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class P_Half_Leg_Left_Right():
    @staticmethod
    def steps():
        return 2
        
    def polyfold_symm_ln(self, left_leg_right, left_leg_left, top_left, top_right, right_leg_right, right_leg_left,crotch, step=0):
        if step == 0:
            return (intercept(perpendicular(make_ln_from_pts(top_left, left_leg_left), pt_center(top_left, left_leg_left)), make_ln_from_pts(left_leg_right, crotch)), pt_center(top_left, left_leg_left))
        elif step == 1:
            return (pt_center(top_right, right_leg_right), intercept(perpendicular(make_ln_from_pts(top_right, right_leg_right), pt_center(top_right, right_leg_right)), make_ln_from_pts(right_leg_left, crotch)))
    
    def oracle_fold(self, left_leg_right, left_leg_left, top_left, top_right, right_leg_right, right_leg_left,crotch, step=0):
        if step == 0:
            return [(left_leg_left, top_left), (left_leg_right, pt_center(top_left, top_right))]
        elif step == 1:
            return [(right_leg_right, top_right), (right_leg_left, pt_center(top_left, top_right))]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the left part of the pant in half from the bottom left leg edge to the left top waist. Then do the same for the right leg.",
        "Fold the left side of the pants in half from the bottom left leg edge up to the left top waist. Then do the same for the right leg.",
        "Bring the bottom left leg edge of the pants up to the left top waist, folding the left side in half. Repeat the same for the right leg.",
        "Fold the left part of the pants in half, starting from the bottom left leg edge and extending up to the left top waist. Then do the same for the right leg.",
        "Position the bottom left leg edge of the pants to meet the left top waist, folding the left part in half. Repeat the same for the right leg."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class P_Half_Leg_Right_Left():
    @staticmethod
    def steps():
        return 2
        
    def polyfold_symm_ln(self, left_leg_right, left_leg_left, top_left, top_right, right_leg_right, right_leg_left,crotch, step=0):
        if step == 0:
            return (pt_center(top_right, right_leg_right), intercept(perpendicular(make_ln_from_pts(top_right, right_leg_right), pt_center(top_right, right_leg_right)), make_ln_from_pts(right_leg_left, crotch)))
        elif step == 1:
            return (intercept(perpendicular(make_ln_from_pts(top_left, left_leg_left), pt_center(top_left, left_leg_left)), make_ln_from_pts(left_leg_right, crotch)), pt_center(top_left, left_leg_left))
    
    def oracle_fold(self, left_leg_right, left_leg_left, top_left, top_right, right_leg_right, right_leg_left,crotch, step=0):
        if step == 0:
            return [(right_leg_right, top_right), (right_leg_left, pt_center(top_left, top_right))]
        elif step == 1:
            return [(left_leg_left, top_left), (left_leg_right, pt_center(top_left, top_right))]
    
    def get_instruction(self, index=-1):
        self.instructions = [
        "Fold the right leg of the pant in half from bottom to top. Then do the same for the left leg.",
        "Fold the right side of the pants in half from the bottom right leg edge up to the right top waist. Then do the same for the left leg.",
        "Bring the bottom right leg edge of the pant up to the right top waist, folding the right side in half. Repeat the same for the left leg.",
        "Fold the right part of the pants in half, starting from the bottom right leg edge and extending up to the right top waist. Then do the same for the left leg.",
        "Position the bottom right leg edge of the pants to meet the right top waist, folding the right part in half. Repeat the same for the left leg."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class P_Block_Left_Right_Bottom_Top():
    @staticmethod
    def steps():
        return 2
        
    def polyfold_symm_ln(self, left_leg_right, left_leg_left, top_left, top_right, right_leg_right, right_leg_left,crotch, step=0):
        if step == 0:
            return (pt_center(left_leg_left, right_leg_right), pt_center(top_left, top_right))
        elif step == 1:
            return (pt_center(top_right, right_leg_right), intercept(perpendicular(make_ln_from_pts(top_right, right_leg_right), pt_center(top_right, right_leg_right)), make_ln_from_pts(right_leg_left, crotch)))
            
    def oracle_fold(self, left_leg_right, left_leg_left, top_left, top_right, right_leg_right, right_leg_left,crotch, step=0):
        if step == 0:
            return [(top_left, top_right), (left_leg_left, right_leg_right)]
        elif step == 1:
            
            return [(right_leg_right, top_right), (right_leg_left, pt_center(top_left, top_right))]
            # top_left_right_center = pt_center(top_left, top_right)
            # scale = 25
            # return [(right_leg_right, top_right), (pt_sum(right_leg_left, pt_scale(line_vector(make_ln_from_pts(right_leg_right, right_leg_left)), scale)), pt_sum(top_left_right_center, pt_scale(line_vector(make_ln_from_pts(top_right, top_left_right_center)), scale)))]

            # return [(pt_center(right_leg_left, right_leg_right), pt_center(top_right, pt_center(top_left, top_right))), (pt_center(right_leg_left, right_leg_right), pt_center(top_right, pt_center(top_left, top_right)))]
    
    def get_instruction(self, index=-1):
        self.instructions = [
            # "Fold the pant into a rectangular block in two steps."
            "Fold the pant in half from left to right and then fold the bottom edge of the pant upwards to meet the top.",
            "First, fold the pants in half horizontally from left to right. Then, fold the bottom edge of the pants upwards to meet the top waist by bending the right bottom right corner to the top right waist.",
            "Begin by folding the pants in half from left to right. Afterward, fold the bottom of the pants upwards, bringing it to meet the top.",
            "Fold the pants horizontally in half, starting from the left and ending at the right. Next, fold the bottom of the pants upwards to meet the top.",
            "Start by folding the pants in half from left to right. Then, fold the lower part of the pants upwards, aligning it with the top."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class P_Block_Right_Left_Top_Bottom():
    @staticmethod
    def steps():
        return 2
        
    def polyfold_symm_ln(self, left_leg_right, left_leg_left, top_left, top_right, right_leg_right, right_leg_left,crotch, step=0):
        if step == 0:
            return (pt_center(top_left, top_right), pt_center(left_leg_left, right_leg_right))
        elif step == 1:
             return (pt_center(top_left, left_leg_left), intercept(perpendicular(make_ln_from_pts(top_left, left_leg_left), pt_center(top_left, left_leg_left)), make_ln_from_pts(left_leg_right, crotch)))
    
    def oracle_fold(self, left_leg_right, left_leg_left, top_left, top_right, right_leg_right, right_leg_left,crotch, step=0):
        if step == 0:
            return [(top_right, top_left), (right_leg_right, left_leg_left)]
        elif step == 1:
            return [(top_left, left_leg_left), (pt_center(top_left, top_right), left_leg_right)]
    
    def get_instruction(self, index=-1):
        self.instructions = [
            "Fold the pant in half from right to left. Then fold the top of the pant downwards to meet the bottom.",
            "First, fold the pants in half horizontally from right to left. Then, fold the top waist of the pants downwards to meet the bottom edge by bending the top right waist to the right bottom right corner."
            # "First, fold the pants in half horizontally from right to left. Then, fold the top of the pants downwards to meet the bottom.",
            "Begin by folding the pants in half from right to left. Afterward, fold the top of the pants downwards, bringing it to meet the bottom.",
            "Fold the pants horizontally in half, starting from the right and ending at the left. Next, fold the top of the pants downwards to meet the bottom.",
            "Start by folding the pants in half from right to left. Then, fold the upper part of the pants downwards, aligning it with the bottom."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class P_Block_Left_Right_Top_Bottom():
    @staticmethod
    def steps():
        return 2
        
    def polyfold_symm_ln(self, left_leg_right, left_leg_left, top_left, top_right, right_leg_right, right_leg_left,crotch, step=0):
        if step == 0:
            return (pt_center(left_leg_left, right_leg_right), pt_center(top_left, top_right))
        elif step == 1:
            return (intercept(perpendicular(make_ln_from_pts(top_right, right_leg_right), pt_center(top_right, right_leg_right)), make_ln_from_pts(right_leg_left, crotch)), pt_center(top_right, right_leg_right))
    
    def oracle_fold(self, left_leg_right, left_leg_left, top_left, top_right, right_leg_right, right_leg_left,crotch, step=0):
        if step == 0:
            return [(top_left, top_right), (left_leg_left, right_leg_right)]
        elif step == 1:
            return [(top_right, right_leg_right), (pt_center(top_left, top_right), right_leg_left)]
    
    def get_instruction(self, index=-1):
        self.instructions = [
            "Fold the pant in half from left to right. Then fold the top of the pant downwards to meet the bottom.",
            "First, fold the pants in half horizontally from left to right. Then, fold the top waist of the pants downwards to meet the bottom edge by bending the top right waist to the right bottom right corner."
            # "First, fold the pants in half horizontally from left to right. Then, fold the top of the pants downwards to meet the bottom.",
            "Begin by folding the pants in half from left to right. Afterward, fold the top of the pants downwards, bringing it to meet the bottom.",
            "Fold the pants horizontally in half, starting from the left and ending at the right. Next, fold the top of the pants downwards to meet the bottom.",
            "Start by folding the pants in half from left to right. Then, fold the upper part of the pants downwards, aligning it with the bottom."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]

class P_Block_Right_Left_Bottom_Top():
    @staticmethod
    def steps():
        return 2
        
    def polyfold_symm_ln(self, left_leg_right, left_leg_left, top_left, top_right, right_leg_right, right_leg_left,crotch, step=0):
        if step == 0:
            return (pt_center(top_right, top_left), pt_center(left_leg_left, right_leg_right))
        elif step == 1:
            return (intercept(perpendicular(make_ln_from_pts(top_left, left_leg_left), pt_center(top_left, left_leg_left)), make_ln_from_pts(left_leg_right, crotch)), pt_center(top_left, left_leg_left))
        
    def oracle_fold(self, left_leg_right, left_leg_left, top_left, top_right, right_leg_right, right_leg_left,crotch, step=0):
        if step == 0:
            return [(top_right, top_left), (right_leg_right, left_leg_left)]
        elif step == 1:
            return [(left_leg_left, top_left), (left_leg_right, pt_center(top_left, top_right))]
    
    def get_instruction(self, index=-1):
        self.instructions = [
            "Fold the pant in half from right to left. Then fold the bottom of the pant upwards to meet the top.",
            "First, fold the pants in half horizontally from right to left. Then, fold the bottom edge of the pants upwards to meet the top waist by bending the right bottom right corner to the top right waist."
            # "First, fold the pants in half horizontally from right to left. Then, fold the bottom of the pants upwards to meet the top.",
            "Begin by folding the pants in half from right to left. Afterward, fold the bottom of the pants upwards, bringing it to meet the top.",
            "Fold the pants horizontally in half, starting from the right and ending at the left. Next, fold the bottom of the pants upwards to meet the top.",
            "Start by folding the pants in half from right to left. Then, fold the lower part of the pants upwards, aligning it with the top."
        ]
        if index == -1:
            return random.choice(self.instructions)
        else:
            return self.instructions[index%len(self.instructions)]
        
        
# ###new
# class S_Corners_Lefttop_Middle():
#     def get_instruction(self, index=-1):
#         self.instructions = [
#         "Fold the left top corner of the square into the center",
#         "Fold the upper-left corner of the square towards the center.",
#         "Take the top-left corner of the square and fold it inwards, towards the center.",
#         "Bring the corner situated at the top-left of the square towards the center by folding.",
#         "Position the square so that its top-left corner aligns with the center, folding it inward."
#         ]
#         if index == -1:
#             return random.choice(self.instructions)
#         else:
#             return self.instructions[index%len(self.instructions)]

# class S_Corners_Lefttop_Rightbottom_Middle():
#     def get_instruction(self, index=-1):
#         self.instructions = [
#         "Fold the left top and right bottom corners of the square into the center.",
#         "Fold both the top-left and bottom-right corners of the square towards the center.",
#         "Bring the corners at the top-left and bottom-right of the square towards the center and fold them inward.",
#         "Fold inward the corners positioned at the top-left and bottom-right of the square, bringing them towards the center.",
#         "Position the square so that both the top-left and bottom-right corners align with the center, folding them inward."
#         ]
#         if index == -1:
#             return random.choice(self.instructions)
#         else:
#             return self.instructions[index%len(self.instructions)]

# class S_Corners_All_Middle():
#     def get_instruction(self, index=-1):
#         self.instructions = [
#         "Fold all corners of the square into the center.",
#         "Fold all corners of the square towards the center.",
#         "Bring all corners of the square towards the center and fold them inward.",
#         "Fold inward all corners of the square, bringing them towards the center.",
#         "Position the square so that all corners align with the center, folding them inward in 4 steps."
#         ]
#         if index == -1:
#             return random.choice(self.instructions)
#         else:
#             return self.instructions[index%len(self.instructions)]

# class S_Triangles_Double_Lefttop_Rightbottom_Righttop_Leftbottom():
#     def get_instruction(self, index=-1):
#         self.instructions = [
#         "Fold the top-left corner towards the bottom-right corner, and fold the top-right corner towards the bottom-left corner.",
#         "Bring the top-left corner down to meet the bottom-right corner, then fold the top-right corner down to meet the bottom-left corner.",
#         "Fold the top-left corner diagonally to meet the bottom-right corner, and similarly, fold the top-right corner diagonally to meet the bottom-left corner.",
#         "Position the square such that the top-left corner folds over to the bottom-right corner, and the top-right corner folds over to the bottom-left corner.",
#         "Converge the top-left corner towards the bottom-right corner, then bring the top-right corner downwards to meet the bottom-left corner."
#         ]
#         if index == -1:
#             return random.choice(self.instructions)
#         else:
#             return self.instructions[index%len(self.instructions)]
     
# class R_Opposite_Top_Down_Middle():
#     def get_instruction(self, index=-1):
#         self.instructions = [
#         "Fold the top-left-top-right line of the rectangle downwards to the horizontal middle-left-middle-right line of the rectangle, and then fold the bottom-left-bottom-right line of the rectangle upwards to the horizontal middle-left-middle-right line of the rectangle.",
#         "Fold the line from the top-left corner to the top-right corner downwards, aligning it with the horizontal line from the middle-left to the middle-right of the rectangle, then fold the line from the bottom-left corner to the bottom-right corner upwards, aligning it with the horizontal line from the middle-left to the middle-right of the rectangle.",
#         "Bring the line extending from the top-left corner to the top-right corner down to meet the horizontal line spanning from the middle-left to the middle-right of the rectangle, and then bring the line extending from the bottom-left corner to the bottom-right corner up to meet the horizontal line spanning from the middle-left to the middle-right of the rectangle.",
#         "Fold the top-left to top-right line of the rectangle downwards, aligning it with the horizontal line drawn from the middle-left to the middle-right of the rectangle, and then fold the bottom-left to bottom-right line of the rectangle upwards, aligning it with the horizontal line drawn from the middle-left to the middle-right of the rectangle.",
#         "Position the top-left to top-right line of the rectangle downwards, meeting it with the horizontal line stretching from the middle-left to the middle-right of the rectangle, and then position the bottom-left to bottom-right line of the rectangle upwards, meeting it with the horizontal line stretching from the middle-left to the middle-right of the rectangle."
#         ]
#         if index == -1:
#             return random.choice(self.instructions)
#         else:
#             return self.instructions[index%len(self.instructions)]

# class R_Edges_Half_Top_Down():
#     def get_instruction(self, index=-1):
#         self.instructions = [
#         "Fold the top edge of the rectangle to the bottom edge.",
#         "Bring the top side of the rectangle down to meet the bottom edge.",
#         "Take the top edge of the rectangle and fold it to the bottom edge.",
#         "Converge the top border of the rectangle towards the bottom edge by folding.",
#         "Fold the top edge of the rectangle towards the bottom edge, aligning them."
#         ]
#         if index == -1:
#             return random.choice(self.instructions)
#         else:
#             return self.instructions[index%len(self.instructions)]

# class R_Edges_Double_Half_Top_Bottom_Left_Right():
#     def get_instruction(self, index=-1):
#         self.instructions = [
#         "Fold the top edge of the rectangle to the bottom edge, then fold the left edge to the right edge.",
#         "Bring the top side of the rectangle down to meet the bottom edge, then fold the left side to the right edge.",
#         "Take the top edge of the rectangle and fold it to the bottom edge, then fold the left edge to the right edge.",
#         "Converge the top border of the rectangle towards the bottom edge by folding, then fold the left border to the right edge.",
#         "Fold the top edge of the rectangle towards the bottom edge, aligning them, then fold the left edge towards the right edge."
#         ]
#         if index == -1:
#             return random.choice(self.instructions)
#         else:
#             return self.instructions[index%len(self.instructions)]

# class T_LeftSleeve():
#     def get_instruction(self, index=-1):
#         self.instructions = [
#         "Fold the left sleeve of the tshirt into the center with the left armpit-shoulder line as the symmetrical axis.",
#         "Bring the left sleeve of the shirt inwards towards the center.",
#         "Converge the left sleeve of the shirt towards the center by folding it inward.",
#         "Position the shirt so that the left sleeve is folded inwardly, moving towards the center",
#         "Fold the left sleeve inward, directing it towards the center of the shirt with the left armpit-shoulder line as the symmetrical axis."
#         ]
#         if index == -1:
#             return random.choice(self.instructions)
#         else:
#             return self.instructions[index%len(self.instructions)]

# class T_Left_Right_Sleeves():
#     def get_instruction(self, index=-1):
#         self.instructions = [
#         "Fold both the left and right sleeves of the tshirt into the center with the armpit-shoulder lines as the symmetrical axes.",
#         "Bring both the left and right sleeves of the shirt inwards towards the center.",
#         "Converge both the left and right sleeves of the shirt towards the center by folding them inward.",
#         "Position the shirt so that both the left and right sleeves are folded inwardly, moving towards the center",
#         "Fold both the left and right sleeves inward, directing them towards the center of the shirt with the armpit-shoulder lines as the symmetrical axes."
#         ]
#         if index == -1:
#             return random.choice(self.instructions)
#         else:
#             return self.instructions[index%len(self.instructions)]


