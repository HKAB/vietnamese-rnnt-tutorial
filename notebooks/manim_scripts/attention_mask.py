from manim import *
import numpy as np

class AttentionMask(Scene):
    def construct(self):
        TIME = 16
        ATTENTION_CONTEXT_SIZE = [4, 1]

        label = MathTex(r"\text{Attention Weight Matrix}").scale(1.25)
        label.to_edge(UP)

        # Create a TIME x TIME matrix by Squares
        squares = VGroup(*[Square(side_length=0.3, color=BLUE_A, fill_color=BLUE_A, fill_opacity=0.5) for _ in range(TIME**2)])
        squares.arrange_in_grid(TIME, TIME, buff=0.05).next_to(label, DOWN, buff=0.5)

        square_label = []
        for i in range(TIME):
            square_label.append(Tex(str(i), font_size=24).next_to(squares[i], UP, buff=0.1))
            square_label.append(Tex(str(i), font_size=24).next_to(squares[i*TIME], LEFT, buff=0.1))
        square_label = VGroup(*square_label)

        chunk_info = VGroup([
                Tex("CHUNK SIZE = 2", font_size=24),
                Tex("LOOK BACK: 2 chunk", font_size=24)
        ]).arrange(DOWN).next_to(squares, RIGHT, buff=1.0)
        
        self.play(Write(label))
        self.play(Create(squares), Write(square_label), Write(chunk_info))
        
        for i in range(0, TIME, 2):
            self.play(FadeToColor(VGroup([squares[i*TIME + j] for j in range(TIME) if j not in range(i - ATTENTION_CONTEXT_SIZE[0], i + ATTENTION_CONTEXT_SIZE[1] + 1)]), color=LIGHT_GREY, fill_color=LIGHT_GREY, fill_opacity=0.5))
            self.play(FadeToColor(VGroup([squares[(i + 1)*TIME + j] for j in range(TIME) if j not in range(i - ATTENTION_CONTEXT_SIZE[0], i + ATTENTION_CONTEXT_SIZE[1] + 1)]), color=LIGHT_GREY, fill_color=LIGHT_GREY, fill_opacity=0.5))
        
        self.wait(2)

        square_group_1 = VGroup([
            Square(side_length=0.6, color=BLUE_A, fill_color=BLUE_A, fill_opacity=0.5),
            Square(side_length=0.6, color=BLUE_A, fill_color=BLUE_A, fill_opacity=0.5),
        ]).arrange(RIGHT, buff=0.05)

        square_group_2 = VGroup([
            Square(side_length=0.6, color=BLUE_A, fill_color=BLUE_A, fill_opacity=0.5),
            Square(side_length=0.6, color=BLUE_A, fill_color=BLUE_A, fill_opacity=0.5),
        ]).arrange(RIGHT, buff=0.05)

        history_group = VGroup([square_group_1, square_group_2]).arrange(RIGHT, buff=0.4)

        square_group_3 = VGroup([
            Square(side_length=0.6, color=RED, fill_color=BLUE_A, fill_opacity=0.5),
            Square(side_length=0.6, color=BLUE_A, fill_color=BLUE_A, fill_opacity=0.5),
        ]).arrange(RIGHT, buff=0.05)

        square_6 = VGroup([history_group, square_group_3]).arrange(RIGHT, buff=0.4)

        chunk_size_brace_label = BraceLabel(square_group_1, "chunk\_size", brace_direction=UP, buff=0.5)
        history_brace_label = BraceLabel(history_group, "history", brace_direction=DOWN, buff=0.5)
        current_frame_brace_label = BraceLabel(square_group_3[0], "current\_frame", brace_direction=UP, buff=1.0)
        look_ahead_frame_brace_label = BraceLabel(square_group_3[1], "look\_ahead", brace_direction=DOWN, buff=1.0)

        square_group_1_label = VGroup([Tex(str(i), font_size=24).next_to(square_group_1[i - 2], UP, buff=0.1) for i in range(2, 4)])
        first_square_label = Tex("6", font_size=24).next_to(square_group_1[0], LEFT, buff=0.2)
        square_group_2_label = VGroup([Tex(str(i), font_size=24).next_to(square_group_2[i - 4], UP, buff=0.1) for i in range(4, 6)])
        square_group_3_label = VGroup([Tex(str(i), font_size=24).next_to(square_group_3[i - 6], UP, buff=0.1) for i in range(6, 8)])

        self.play(FadeOut(VGroup(*[squares[i*TIME + j] for j in range(TIME) for i in range(TIME) if i != 6])))
        self.play(FadeOut(chunk_info))

        self.wait(1)

        self.play(FadeTransform(square_label, VGroup([square_group_1_label, square_group_2_label, square_group_3_label, first_square_label])),
                  FadeTransform(VGroup(*[squares[6*TIME + i] for i in range(TIME)]), square_6))

        self.play(Create(chunk_size_brace_label))
        self.play(Create(history_brace_label))
        self.play(Create(current_frame_brace_label))
        self.play(Create(look_ahead_frame_brace_label))

        self.wait(1)
        