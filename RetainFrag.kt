package com.ninhydrin.doodly

import android.app.Fragment
import android.os.Bundle

class RetainFrag : Fragment() {
    private var presenter: Presenter? = null

    fun setPresenter(p: Presenter) {
        presenter = p
    }

    fun getPresenter() = presenter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        retainInstance = true
    }
}