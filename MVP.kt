package com.ninhydrin.doodly

class MVP {

    interface ViewToPresenter {
        fun donePressed(pixels:IntArray)
    }

    interface PresenterToModel {
        fun done(pixels: IntArray)
    }

    interface ModelToPresenter {
        fun passPrediction(prediction: String)
    }

    interface PresenterToView {
        fun showResult(result: String)
    }

}