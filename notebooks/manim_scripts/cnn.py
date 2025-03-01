from manim import *
import numpy as np

class CNNStreaming(Scene):
    def construct(self):
        MEL_BLOCK = 33

        # STEP 1
        # Mel Spectrogram Representation
        mel_label = Tex("Mel Spectrogram", font_size=24).to_edge(UP, buff=0.5)
        mel_spectrogram = VGroup(*[Square(side_length=0.3, color=BLUE_A, fill_opacity=0.5) 
                                   for _ in range(MEL_BLOCK)]).arrange(RIGHT, buff=0.1).next_to(mel_label, DOWN, buff=0.6)
        mel_idx = VGroup(*[
            MathTex(r"{" + str(i) + r"}", font_size=16).next_to(mel_spectrogram[i], UP, buff=0.2) for i in range(MEL_BLOCK)]
        )
        mel_spectrogram[0].color = YELLOW_A
        
        conv1_kernel_slides = VGroup(*[
            SurroundingRectangle(mel_spectrogram[i:i + 3], buff=0.1) for i in range(0, MEL_BLOCK - 3 + 1, 2)
        ])
        
        CONV1_BLOCK = (MEL_BLOCK + 2*0 - 1 * (3 - 1) - 1) // 2 + 1
        CONV1_BLOCK = CONV1_BLOCK + 1 # Cache 1
        conv1_label = Tex("Conv1 output", font_size=24).next_to(mel_spectrogram, DOWN, buff=0.5)
        conv1 = VGroup(*[Square(side_length=0.3, color=BLUE_A, fill_opacity=0.5) 
                                   for _ in range(CONV1_BLOCK)]).arrange(RIGHT, buff=0.1).next_to(conv1_label, DOWN, buff=0.4)
        conv1_idx = VGroup(*[
            MathTex(r"{" + str(i) + r"}", font_size=16).next_to(conv1[i], UP, buff=0.2) for i in range(CONV1_BLOCK)]
        )
        conv1[0].color = YELLOW_A

        conv1_kernel_result = VGroup(*[
            Arrow(conv1_kernel_slides[i].get_bottom(), conv1[i + 1].get_center(), buff=0.1) for i in range(len(conv1_kernel_slides))
        ])

        conv2_kernel_slides = VGroup(*[
            SurroundingRectangle(conv1[i:i + 3], buff=0.1) for i in range(0, CONV1_BLOCK - 3 + 1, 2)
        ])

        CONV2_BLOCK = (CONV1_BLOCK + 2*0 - 1 * (3 - 1) - 1) // 2 + 1
        CONV2_BLOCK = CONV2_BLOCK + 1 # Cache 1
        conv2_label = Tex("Conv2 output", font_size=24).next_to(conv1, DOWN, buff=0.5)
        conv2 = VGroup(*[Square(side_length=0.3, color=BLUE_A, fill_opacity=0.5) 
                                   for _ in range(CONV2_BLOCK)]).arrange(RIGHT, buff=0.1).next_to(conv2_label, DOWN, buff=0.4)
        conv2_idx = VGroup(*[
            MathTex(r"{" + str(i) + r"}", font_size=16).next_to(conv2[i], UP, buff=0.2) for i in range(CONV2_BLOCK)]
        )
        conv2[0].color = YELLOW_A

        conv2_kernel_result = VGroup(*[
            Arrow(conv2_kernel_slides[i].get_bottom(), conv2[i + 1].get_center(), buff=0.1) for i in range(len(conv2_kernel_slides))
        ])

        conv3_kernel_slides = VGroup(*[
            SurroundingRectangle(conv2[i:i + 3], buff=0.1) for i in range(0, CONV2_BLOCK - 3 + 1, 2)
        ])

        CONV3_BLOCK = (CONV2_BLOCK + 2*0 - 1 * (3 - 1) - 1) // 2 + 1
        conv3_label = Tex("Conv3 output", font_size=24).next_to(conv2, DOWN, buff=0.5)
        conv3 = VGroup(*[Square(side_length=0.3, color=BLUE_A, fill_opacity=0.5) 
                                   for _ in range(CONV3_BLOCK)]).arrange(RIGHT, buff=0.1).next_to(conv3_label, DOWN, buff=0.5)
        conv3_idx = VGroup(*[
            MathTex(r"{" + str(i) + r"}", font_size=16).next_to(conv3[i], UP, buff=0.2) for i in range(CONV3_BLOCK)]
        )

        conv3_kernel_result = VGroup(*[
            Arrow(conv3_kernel_slides[i].get_bottom(), conv3[i].get_center(), buff=0.1) for i in range(len(conv3_kernel_slides))
        ])

        encoder_output_label = Tex("Subsampling result", font_size=24).next_to(conv3, DOWN, buff=0.5)
        encoder_output_result = VGroup([Square(side_length=0.3, color=BLUE_A, fill_opacity=0.5) for i in range(0, 8)]).arrange(RIGHT, buff=0.1).next_to(encoder_output_label, DOWN, buff=0.5)
        encoder_output_idx = VGroup([MathTex(r"{" + str(i) + r"}", font_size=16).next_to(encoder_output_result[i], UP, buff=0.2) for i in range(8)])

        self.play(Create(mel_spectrogram), Write(mel_idx), Write(mel_label), Write(conv1_label), Create(conv1[0]), Write(conv1_idx[0]), Write(conv2_label), Create(conv2[0]), Write(conv2_idx[0]), Write(conv3_label), Write(encoder_output_label))
        # self.play(Create(conv1_kernel_slides), Create(conv1), Write(conv1_idx), Create(conv1_kernel_result))
        self.play(Create(conv1_kernel_slides[0]))
        self.play(Create(conv1_kernel_result[0]))
        self.play(Create(conv1[1]), Write(conv1_idx[1]))
        for i in range(1, len(conv1_kernel_slides)):
            self.play(ReplacementTransform(conv1_kernel_slides[i - 1], conv1_kernel_slides[i], lag_ratio = 0.5))
            self.play(Create(conv1_kernel_result[i]), Create(conv1[i + 1]), Write(conv1_idx[i + 1]))
            self.play(FadeOut(conv1_kernel_result[i - 1]))
        self.play(FadeOut(conv1_kernel_result[-1]), FadeOut(conv1_kernel_slides[-1]))

        self.play(Create(conv2_kernel_slides[0]))
        self.play(Create(conv2_kernel_result[0]))
        self.play(Create(conv2[1]), Write(conv2_idx[1]))
        for i in range(1, len(conv2_kernel_slides)):
            self.play(ReplacementTransform(conv2_kernel_slides[i - 1], conv2_kernel_slides[i]))
            self.play(Create(conv2_kernel_result[i]), Create(conv2[i + 1]), Write(conv2_idx[i + 1]))
            self.play(FadeOut(conv2_kernel_result[i - 1]))
        self.play(FadeOut(conv2_kernel_result[-1]), FadeOut(conv2_kernel_slides[-1]))

        self.play(Create(conv3_kernel_slides[0]))
        self.play(Create(conv3_kernel_result[0]))
        self.play(Create(conv3[0]), Write(conv3_idx[0]))

        for i in range(1, len(conv3_kernel_slides)):
            self.play(ReplacementTransform(conv3_kernel_slides[i - 1], conv3_kernel_slides[i]))
            self.play(Create(conv3_kernel_result[i]), Create(conv3[i]), Write(conv3_idx[i]))
            self.play(FadeOut(conv3_kernel_result[i - 1]))
        self.play(FadeOut(conv3_kernel_result[-1]), FadeOut(conv3_kernel_slides[-1]))

        # STEP 2
        self.play(FadeOut(*[mel_spectrogram[i] for i in range(1, MEL_BLOCK - 1)]), FadeOut(*[mel_idx[i] for i in range(1, MEL_BLOCK - 1)]))
        self.play(Swap(mel_spectrogram[-1], mel_spectrogram[0]), Swap(mel_idx[-1], mel_idx[0]), FadeOut(mel_spectrogram[0]), FadeOut(mel_idx[0]))
        self.play(FadeOut(*[conv1[i] for i in range(1, CONV1_BLOCK - 1)]), FadeOut(*[conv1_idx[i] for i in range(1, CONV1_BLOCK - 1)]))
        self.play(Swap(conv1[-1], conv1[0]), Swap(conv1_idx[-1], conv1_idx[0]), FadeOut(conv1[0]), FadeOut(conv1_idx[0]))
        self.play(FadeOut(*[conv2[i] for i in range(1, CONV2_BLOCK - 1)]), FadeOut(*[conv2_idx[i] for i in range(1, CONV2_BLOCK - 1)]))
        self.play(Swap(conv2[-1], conv2[0]), Swap(conv2_idx[-1], conv2_idx[0]), FadeOut(conv2[0]), FadeOut(conv2_idx[0]))
        # self.play(FadeOut(*[conv3[i] for i in range(CONV3_BLOCK)]), FadeOut(*[conv3_idx[i] for i in range(CONV3_BLOCK)]))
        self.play(FadeTransform(conv3, VGroup([encoder_output_result[i] for i in range(4)])), FadeTransform(conv3_idx, VGroup([encoder_output_idx[i] for i in range(4)])))

        mel_spectrogram = VGroup([mel_spectrogram[-1]] + [Square(side_length=0.3, color=BLUE_A, fill_opacity=0.5) 
                                   for _ in range(MEL_BLOCK - 1)]).arrange(RIGHT, buff=0.1).next_to(mel_label, DOWN, buff=0.6)
        mel_idx = VGroup([mel_idx[-1]] + [
            MathTex(r"{" + str(MEL_BLOCK + i) + r"}", font_size=16).next_to(mel_spectrogram[i + 1], UP, buff=0.2) for i in range(MEL_BLOCK - 1)]
        )

        conv1 = VGroup([conv1[-1]] + [Square(side_length=0.3, color=BLUE_A, fill_opacity=0.5)
                                      for _ in range(CONV1_BLOCK - 1)]).arrange(RIGHT, buff=0.1).next_to(conv1_label, DOWN, buff=0.5)
        conv1_idx = VGroup([conv1_idx[-1]] + [
            MathTex(r"{" + str(CONV1_BLOCK + i) + r"}", font_size=16).next_to(conv1[i + 1], UP, buff=0.2) for i in range(CONV1_BLOCK - 1)]
        )

        conv2 = VGroup([conv2[-1]] + [Square(side_length=0.3, color=BLUE_A, fill_opacity=0.5)
                                        for _ in range(CONV2_BLOCK - 1)]).arrange(RIGHT, buff=0.1).next_to(conv2_label, DOWN, buff=0.5)
        conv2_idx = VGroup([conv2_idx[-1]] + [
            MathTex(r"{" + str(CONV2_BLOCK + i) + r"}", font_size=16).next_to(conv2[i + 1], UP, buff=0.2) for i in range(CONV2_BLOCK - 1)]
        )

        conv3 = VGroup(*[Square(side_length=0.3, color=BLUE_A, fill_opacity=0.5)
                                        for _ in range(CONV3_BLOCK)]).arrange(RIGHT, buff=0.1).next_to(conv3_label, DOWN, buff=0.5)
        
        conv3_idx = VGroup(*[
            MathTex(r"{" + str(CONV3_BLOCK + i) + r"}", font_size=16).next_to(conv3[i], UP, buff=0.2) for i in range(CONV3_BLOCK)]
        )

        conv1_kernel_slides = VGroup(*[
            SurroundingRectangle(mel_spectrogram[i:i + 3], buff=0.1) for i in range(0, MEL_BLOCK - 3 + 1, 2)
        ])

        conv1_kernel_result = VGroup(*[
            Arrow(conv1_kernel_slides[i].get_bottom(), conv1[i + 1].get_center(), buff=0.1) for i in range(len(conv1_kernel_slides))
        ])

        conv2_kernel_slides = VGroup(*[
            SurroundingRectangle(conv1[i:i + 3], buff=0.1) for i in range(0, CONV1_BLOCK - 3 + 1, 2)
        ])

        conv2_kernel_result = VGroup(*[
            Arrow(conv2_kernel_slides[i].get_bottom(), conv2[i + 1].get_center(), buff=0.1) for i in range(len(conv2_kernel_slides))
        ])

        conv3_kernel_slides = VGroup(*[
            SurroundingRectangle(conv2[i:i + 3], buff=0.1) for i in range(0, CONV2_BLOCK - 3 + 1, 2)
        ])

        conv3_kernel_result = VGroup(*[
            Arrow(conv3_kernel_slides[i].get_bottom(), conv3[i].get_center(), buff=0.1) for i in range(len(conv3_kernel_slides))
        ])


        self.play(
            FadeToColor(mel_spectrogram[0], color=YELLOW_A), FadeToColor(conv1[0], color=YELLOW_A), FadeToColor(conv2[0], color=YELLOW_A),
            Create(mel_spectrogram[1:]), Write(mel_idx[1:])
        )
        # self.play(FadeToColor(conv1[0], color=YELLOW_A), Create(conv1[1:]), Write(conv1_idx[1:]))
        # self.play(FadeToColor(conv2[0], color=YELLOW_A), Create(conv2[1:]), Write(conv2_idx[1:]))
        # self.play(Create(conv3), Write(conv3_idx))

        self.play(Create(conv1_kernel_slides[0]))
        self.play(Create(conv1_kernel_result[0]))
        self.play(Create(conv1[1]), Write(conv1_idx[1]))
        for i in range(1, len(conv1_kernel_slides)):
            self.play(ReplacementTransform(conv1_kernel_slides[i - 1], conv1_kernel_slides[i]))
            self.play(Create(conv1_kernel_result[i]), Create(conv1[i + 1]), Write(conv1_idx[i + 1]))
            self.play(FadeOut(conv1_kernel_result[i - 1]))
        self.play(FadeOut(conv1_kernel_result[-1]), FadeOut(conv1_kernel_slides[-1]))

        self.play(Create(conv2_kernel_slides[0]))
        self.play(Create(conv2_kernel_result[0]))
        self.play(Create(conv2[1]), Write(conv2_idx[1]))
        for i in range(1, len(conv2_kernel_slides)):
            self.play(ReplacementTransform(conv2_kernel_slides[i - 1], conv2_kernel_slides[i]))
            self.play(Create(conv2_kernel_result[i]), Create(conv2[i + 1]), Write(conv2_idx[i + 1]))
            self.play(FadeOut(conv2_kernel_result[i - 1]))
        self.play(FadeOut(conv2_kernel_result[-1]), FadeOut(conv2_kernel_slides[-1]))

        self.play(Create(conv3_kernel_slides[0]))
        self.play(Create(conv3_kernel_result[0]))
        self.play(Create(conv3[0]), Write(conv3_idx[0]))

        for i in range(1, len(conv3_kernel_slides)):
            self.play(ReplacementTransform(conv3_kernel_slides[i - 1], conv3_kernel_slides[i]))
            self.play(Create(conv3_kernel_result[i]), Create(conv3[i]), Write(conv3_idx[i]))
            self.play(FadeOut(conv3_kernel_result[i - 1]))
        self.play(FadeOut(conv3_kernel_result[-1]), FadeOut(conv3_kernel_slides[-1]))
        self.play(FadeTransform(conv3, VGroup([encoder_output_result[i] for i in range(4, 8)])), FadeTransform(conv3_idx, VGroup([encoder_output_idx[i] for i in range(4, 8)])))