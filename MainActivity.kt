package com.ninhydrin.doodly

import android.app.FragmentManager
import android.os.Bundle
import android.support.v7.app.AppCompatActivity
import android.util.Log
import android.view.View
import android.widget.Toast
import kotlinx.android.synthetic.main.activity_main.*
import org.tensorflow.contrib.android.TensorFlowInferenceInterface
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader

@Suppress("UNUSED_PARAMETER")
class MainActivity : AppCompatActivity(), MVP.PresenterToView {

    private val RETAIN_FRAGMENT_TAG = "RetainFragTag"
    private var presenter: Presenter? = null
    private var viewToPresenter: MVP.ViewToPresenter? = null
    lateinit private var fragment_manager: FragmentManager
    private val params: IntArray = IntArray(784)
    lateinit private var inferenceInterface: TensorFlowInferenceInterface

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // initialise all the variables
        initVariables()

        // initialise the presenter
        initPresenter()
        getParams()

    }

    private fun initVariables() {
        fragment_manager = fragmentManager

        // get the params
        getParams()

        inferenceInterface = TensorFlowInferenceInterface(assets, "final_output.pb")

    }

    private fun initPresenter() {
        var retainFrag: RetainFrag? = fragment_manager.findFragmentByTag(RETAIN_FRAGMENT_TAG)
                as RetainFrag?

        if (retainFrag == null) {
            retainFrag = RetainFrag()
            fragment_manager.beginTransaction().add(retainFrag, RETAIN_FRAGMENT_TAG).commit()

            presenter = Presenter(inferenceInterface)
            retainFrag.setPresenter(presenter as Presenter)
        } else
            presenter = retainFrag.getPresenter()

        presenter?.attach(this)
        viewToPresenter = presenter?.getViewToPresenter()

    }

    private fun getParams() {
        val inputStream = applicationContext.resources.openRawResource(R.raw.params)
        val inputReader = InputStreamReader(inputStream)
        val bufferedReader = BufferedReader(inputReader)

        try {
            var i = 0
            while (i <= 783) {
                val c = bufferedReader.read()
                if (c != 13 && c != 10) {
                    params[i] = c - 48
                    i++
                }
            }
            Log.w("rrr", i.toString())
        } catch (e: IOException) {
            e.printStackTrace()
        } finally {
            bufferedReader.close()
            inputReader.close()
            inputStream.close()
        }
    }


    // clear button pressed
    fun clearDrawing(view: View) {
        customCanvas.clear()
    }

    // done button pressed
    fun doneDrawing(view: View) {
        if (viewToPresenter != null)
            customCanvas.done(viewToPresenter as MVP.ViewToPresenter, params)
        else
            Toast.makeText(applicationContext, "ViewToPresenter null while sending to Custom View",
                    Toast.LENGTH_SHORT).show()
    }


    //--------------------------------presenter->View --------------------------------//

    override fun showResult(result: String) {
        Toast.makeText(applicationContext, result, Toast.LENGTH_SHORT).show()
    }


    // ----------------------------------------------------------------------------------//
    override fun onDestroy() {
        presenter?.detach()
        super.onDestroy()
    }
}
