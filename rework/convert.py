import coremltools

coreml_model = coremltools.convert(
    "exclusive/INPUT_TYPES.Relcom_Mediapipe_DNN_cons_04.h5",
    convert_to="neuralnetwork",
    inputs=[coremltools.TensorType("dense_4_input", shape=(272,))],
    # outputs=[coremltools.TensorType()],
    classifier_config=coremltools.ClassifierConfig(
        [
            "none",
            "wave",
            "drink",
            "watch",
            "none",
            "none",
            "none",
            "none",
            "none",
            "none",
            "none",
        ]
    ),
    source="tensorflow",
)

coreml_model.save("dnn_relcom_cfc_04.mlmodel")
