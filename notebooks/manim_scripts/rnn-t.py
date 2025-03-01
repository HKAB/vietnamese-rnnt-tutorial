from manim import *
import numpy as np

class RNNTVisualization(Scene):
    def construct(self):
        MEL_BLOCK = 8
        ENCODER_HIDDEN_BLOCK = 5
        TEXT_IDS_BLOCK = 6
        DECODER_HIDDEN_BLOCK = 6

        video_label = MathTex(r"\text{RNN-T}").scale(0.8)
        video_label.to_edge(UP)

        # Audio Wave Representation
        wave = FunctionGraph(
            lambda x: 0.1 * np.sin(4 * np.pi * x),
            x_range=[-1, 1],
            color=BLUE
        )
        wave_label = Tex("Audio Wave", font_size=16).next_to(wave, UP)

        # Mel Spectrogram Representation
        mel_spectrogram = VGroup(*[Square(side_length=0.2, color=BLUE_A, fill_opacity=0.5) 
                                   for _ in range(MEL_BLOCK)]).arrange(RIGHT, buff=0.1).next_to(wave_label, UP, buff=0.5)
        mel_label = VGroup(*[
            MathTex(r"m_{" + str(i) + r"}", font_size=16).next_to(mel_spectrogram[i], UP, buff=0.2) for i in range(MEL_BLOCK)]
        )
        

        # Encoder Block
        encoder_block = Rectangle(width=2.0, height=1, color=WHITE, fill_opacity=0.5).next_to(mel_label, UP, buff=0.5)
        encoder_label = Tex("Encoder", font_size=16).move_to(encoder_block)

        # Hidden Representation
        hidden_state = VGroup(*[Square(side_length=0.2, color=BLUE, fill_opacity=0.9) 
                                   for _ in range(ENCODER_HIDDEN_BLOCK)]).arrange(RIGHT, buff=0.2).next_to(encoder_block, UP, buff=0.5)
        hidden_state_label = VGroup(*[
            MathTex(r"f_{" + str(i) + r"}", font_size=16).next_to(hidden_state[i], UP, buff=0.2) for i in range(ENCODER_HIDDEN_BLOCK)]
        )

        encoder_group = VGroup(
            wave, wave_label, mel_spectrogram, mel_label, 
            encoder_block, encoder_label, 
            hidden_state, hidden_state_label).to_corner(DL, buff=5.0)

        # Jointer Block
        jointer_block = Rectangle(width=2.0, height=1, color=WHITE, fill_opacity=0.5)
        jointer_label = Tex("Jointer", font_size=16).move_to(jointer_block)

        jointer_hidden_state = Square(side_length=0.2, color=BLUE, fill_opacity=0.9).next_to(jointer_block, RIGHT, buff=0.5)
        jointer_hidden_state_label = VGroup(*[
            MathTex(r"h_{" + str(i) + r"," + str(i + 1) + r"}", font_size=16).next_to(jointer_hidden_state, UP, buff=0.2)
            for i in range(ENCODER_HIDDEN_BLOCK)
        ])
        # output_argmax = Tex("h", font_size=16).next_to(jointer_hidden_state, RIGHT, buff=0.5)

        output_argmax = VGroup(*[
            Tex("h", font_size=16).next_to(jointer_hidden_state, RIGHT, buff=2.0),
            Tex("e", font_size=16).next_to(jointer_hidden_state, RIGHT, buff=2.0),
            Tex("l", font_size=16).next_to(jointer_hidden_state, RIGHT, buff=2.0),
            Tex("l", font_size=16).next_to(jointer_hidden_state, RIGHT, buff=2.0),
            Tex("o", font_size=16).next_to(jointer_hidden_state, RIGHT, buff=2.0),
            ]
        )

        jointer_group = VGroup(
            jointer_block, jointer_label, 
            jointer_hidden_state, jointer_hidden_state_label, output_argmax
        ).next_to(encoder_group, UP, buff=1.0)

        encoder_jointer_group = VGroup(encoder_group, jointer_group).to_edge(DOWN)

        # Decoder Block
        text_ids = VGroup(*[Square(side_length=0.2, color=BLUE_A, fill_opacity=0.5) 
                                   for _ in range(TEXT_IDS_BLOCK)]).arrange(RIGHT, buff=0.2).next_to(wave_label, UP, buff=0.5)
        text_label = VGroup(*[
            # MathTex(r"c_{" + str(i) + r"}", font_size=16).next_to(mel_spectrogram[i], UP, buff=0.2) for i in range(5)]
            Tex(r"\textless s\textgreater", font_size=16).next_to(text_ids[0], UP, buff=0.2),
            Tex("h", font_size=16).next_to(text_ids[1], UP, buff=0.2),
            Tex("e", font_size=16).next_to(text_ids[2], UP, buff=0.2),
            Tex("l", font_size=16).next_to(text_ids[3], UP, buff=0.2),
            Tex("l", font_size=16).next_to(text_ids[4], UP, buff=0.2),
            Tex("o", font_size=16).next_to(text_ids[5], UP, buff=0.2),
            ]
        )

        # Decoder Block
        decoder_block = Rectangle(width=2.0, height=1, color=WHITE, fill_opacity=0.5).next_to(text_ids, UP, buff=0.5)
        decoder_label = Tex("Decoder", font_size=16).move_to(decoder_block)

        rnn_hidden = VGroup(*[Square(side_length=0.2, color=BLUE, fill_opacity=0.9) 
                                   for _ in range(DECODER_HIDDEN_BLOCK)]).arrange(RIGHT, buff=0.2).next_to(decoder_block, UP, buff=0.5)
        rnn_hidden_label = VGroup(*[
            MathTex(r"g_{" + str(i) + r"}", font_size=16).next_to(rnn_hidden[i], UP, buff=0.2) for i in range(DECODER_HIDDEN_BLOCK)]
        )

        decoder_group = VGroup(
            text_ids, text_label, 
            decoder_block, decoder_label, 
            rnn_hidden, rnn_hidden_label
        ).next_to(encoder_jointer_group, RIGHT)

        encoder_jointer_group.shift(LEFT)
        decoder_group.shift(RIGHT)

        # ARROWS
        mel_to_encoder_arrows = VGroup(
            *[
                Arrow(mel_spectrogram[i].get_bottom(), encoder_block.get_bottom(), buff=0.1, stroke_width=2) for i in range(MEL_BLOCK)
            ]
        )

        encoder_to_hidden_arrows = VGroup(
            *[
                Arrow(encoder_block.get_top(), hidden_state[i].get_bottom(), buff=0.1, stroke_width=2) for i in range(ENCODER_HIDDEN_BLOCK)
            ]
        )

        text_id_to_decoder_arrows = VGroup(
            *[
                Arrow(text_ids[i].get_bottom(), decoder_block.get_bottom(), buff=0.1, stroke_width=2) for i in range(TEXT_IDS_BLOCK)
            ]
        )

        decoder_to_rnn_hidden_arrows = VGroup(
            *[
                Arrow(decoder_block.get_top(), rnn_hidden[i].get_bottom(), buff=0.1, stroke_width=2) for i in range(DECODER_HIDDEN_BLOCK)
            ]
        )

        rnn_hidden_to_decoder_arrows = VGroup(
            *[
                Arrow(rnn_hidden[i].get_bottom(), decoder_block.get_top(), buff=0.1, stroke_width=2) for i in range(DECODER_HIDDEN_BLOCK)
            ]
        )

        rnn_hidden_to_jointer_arrows = VGroup(
            *[
                Arrow(rnn_hidden[i].get_top(), jointer_block.get_bottom(), buff=0.1, stroke_width=2) for i in range(DECODER_HIDDEN_BLOCK)
            ]
        )

        hidden_state_to_jointer_arrows = VGroup(
            *[
                Arrow(hidden_state[i].get_top(), jointer_block.get_bottom(), buff=0.1, stroke_width=2) for i in range(ENCODER_HIDDEN_BLOCK)
            ]
        )

        jointer_to_jointer_hidden_arrow = Arrow(jointer_block.get_right(), jointer_hidden_state.get_left(), buff=0.1, stroke_width=2)
        
        jointer_hidden_state_to_output_argmax_arrow = Arrow(jointer_hidden_state.get_right(), output_argmax[0].get_left(), stroke_width=2)
        softmax_argmax_arrow_label = Tex(r"softmax + argmax", font_size=16).next_to(jointer_hidden_state_to_output_argmax_arrow, UP, buff=0.2)

        jointer_output_argmax_to_text_ids_arrows = VGroup(
            *[
                Arrow(output_argmax.get_bottom(), text_ids[i].get_top(), buff=0.1, stroke_width=2) for i in range(1, TEXT_IDS_BLOCK)
            ]
        )

        wave_to_mel_arrow = Arrow(wave.get_top(), mel_spectrogram.get_bottom(), buff=0.1, stroke_width=2)
        mel_name = Tex("Mel Spectrogram", font_size=16).next_to(mel_spectrogram, LEFT, buff=0.5)
        
        self.play(Write(video_label), Create(encoder_block), Write(encoder_label), Create(decoder_block), Write(decoder_label), Create(jointer_block), Write(jointer_label))

        self.play(Create(wave), Write(wave_label))
        self.play(Write(mel_name), GrowArrow(wave_to_mel_arrow))
        self.play(Create(mel_spectrogram), Write(mel_label))
        self.play(FadeOut(wave_to_mel_arrow))

        self.play(*[GrowArrow(mel_to_encoder_arrows[i]) for i in range(MEL_BLOCK)])
        
        self.play(*[FadeOut(mel_to_encoder_arrows[i]) for i in range(MEL_BLOCK)])

        self.play(* [GrowArrow(encoder_to_hidden_arrows[i]) for i in range(ENCODER_HIDDEN_BLOCK)] + [Create(hidden_state), Write(hidden_state_label)])
        self.play(*[FadeOut(encoder_to_hidden_arrows[i]) for i in range(ENCODER_HIDDEN_BLOCK)])

        self.play(Create(text_ids[0]), Write(text_label[0]), Create(rnn_hidden[0]), Write(rnn_hidden_label[0]))
        for i in range(ENCODER_HIDDEN_BLOCK):
            self.play(GrowArrow(text_id_to_decoder_arrows[i]), GrowArrow(rnn_hidden_to_decoder_arrows[i]))
            self.play(FadeOut(text_id_to_decoder_arrows[i]), FadeOut(rnn_hidden_to_decoder_arrows[i]))
            self.play(GrowArrow(decoder_to_rnn_hidden_arrows[i + 1]), Create(rnn_hidden[i + 1]), Write(rnn_hidden_label[i + 1]))
            self.play(FadeOut(decoder_to_rnn_hidden_arrows[i + 1]))

            self.play(GrowArrow(rnn_hidden_to_jointer_arrows[i + 1]), GrowArrow(hidden_state_to_jointer_arrows[i]))
            self.play(FadeOut(rnn_hidden_to_jointer_arrows[i + 1]), FadeOut(hidden_state_to_jointer_arrows[i]))

            self.play(GrowArrow(jointer_to_jointer_hidden_arrow), Create(jointer_hidden_state), Write(jointer_hidden_state_label[i]))
            self.play(FadeOut(jointer_to_jointer_hidden_arrow))

            self.play(
                GrowArrow(jointer_hidden_state_to_output_argmax_arrow), Write(softmax_argmax_arrow_label), Create(output_argmax[i]))
            self.play(
                FadeOut(jointer_hidden_state), FadeOut(jointer_hidden_state_label[i]),
                FadeOut(jointer_hidden_state_to_output_argmax_arrow), FadeOut(softmax_argmax_arrow_label)
            )

            self.play(
                GrowArrow(jointer_output_argmax_to_text_ids_arrows[i]), 
                Create(text_ids[i+1]), Write(text_label[i+1])
            )
            self.play(
                FadeOut(output_argmax[i]),
                FadeOut(jointer_output_argmax_to_text_ids_arrows[i]), 
                FadeOut(jointer_output_argmax_to_text_ids_arrows[i])
            )

        # self.wait(2)