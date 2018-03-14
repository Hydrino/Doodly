package com.ninhydrin.doodly

import org.tensorflow.contrib.android.TensorFlowInferenceInterface

class Presenter(inferenceInterface: TensorFlowInferenceInterface) : MVP.ViewToPresenter, MVP.ModelToPresenter {

    private val presenterToModel: MVP.PresenterToModel
    private var presenterToView: MVP.PresenterToView? = null
    private val model: Model = Model(this,inferenceInterface)

    init {
        presenterToModel = model.getPresenterToModel()
    }

    // ------------------------------view->Presenter-----------------------------------------//
    override fun donePressed(pixels: IntArray) {
        presenterToModel.done(pixels)
    }


    // --------------------------model->Presenter----------------------------------------------//
    override fun passPrediction(prediction: String) {
        presenterToView?.showResult(prediction)
    }

    // ------------------------lifecycle methods ---------------------------------------------//

    fun attach(p: MVP.PresenterToView) {
        presenterToView = p
    }

    fun detach() {
        presenterToView = null
    }

    // -------------return instance --------------------//
    fun getViewToPresenter() = this

}