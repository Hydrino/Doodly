package com.ninhydrin.doodly

import android.util.Log
import org.tensorflow.contrib.android.TensorFlowInferenceInterface

class Model(private val modelToPresenter: MVP.ModelToPresenter,
            private val inferenceInterface: TensorFlowInferenceInterface) : MVP.PresenterToModel {

    override fun done(pixels: IntArray) {

        val pixels2 = pixels.map { it.toFloat() }.toFloatArray()
        System.out.println("\nModel")
        for (i in 0..27) {
            for (j in 0..27) {
                System.out.print(pixels2[i * 28 + j].toInt())
            }
            System.out.print("\n")
        }
        inferenceInterface.feed("conv2d_1_input", pixels2, 1, 28, 28, 1)
        val outputNames = arrayOf("dense_3/Softmax")
        inferenceInterface.run(outputNames)
        val oo = FloatArray(2)
        inferenceInterface.fetch("dense_3/Softmax", oo)
        val pred = if (oo[0] > oo[1]) "Apple" else "Alarm Clock"
        modelToPresenter.passPrediction(pred + " \n" + oo[0].toString() + " " + oo[1].toString())
    }

    fun getPresenterToModel() = this
}