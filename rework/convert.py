import coremltools

coreml_model = coremltools.convert(
    "exclusive/INPUT_TYPES.Relcom_Mediapipe_DNN_cons_05.h5",
    convert_to="neuralnetwork",
    inputs=[coremltools.TensorType("dense_input", shape=(340,))],
    # outputs=[coremltools.TensorType()],
    classifier_config=coremltools.ClassifierConfig(
        [
            "no_action",
            "falling",
            "drink",
            "placeholder_1",
            "placeholder_2",
            "placeholder_3",
            "quick_sit",
            "placeholder_4",
            "look_at_watch",
            "placeholder_5",
            "placeholder_6",
        ]
    ),
    source="tensorflow",
)

coreml_model.save("dnn_relcom_cfc_05.mlmodel")
