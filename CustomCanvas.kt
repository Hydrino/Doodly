package com.ninhydrin.doodly

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.util.Log
import android.view.MotionEvent
import android.widget.ImageView
import android.widget.Toast

class CustomCanvas(context_: Context, attributeSet: AttributeSet)
    : ImageView(context_, attributeSet) {

    // the Paint object for the doodle
    private val doodlePaint: Paint = Paint(Paint.ANTI_ALIAS_FLAG)

    // the Path object for the doodle
    private val doodlePath: Path = Path()

    // the color of the doodle
    private val doodleColor = Color.argb(255, 255, 255, 255)

    init {
        // important to get bitmap of the doodle
        isDrawingCacheEnabled = true
        // set doodle paint properties
        doodlePaint.color = doodleColor
        doodlePaint.strokeWidth = 64f
        doodlePaint.style = Paint.Style.STROKE
        doodlePaint.strokeJoin = Paint.Join.ROUND
        doodlePaint.strokeCap = Paint.Cap.ROUND
    }

    override fun onSizeChanged(w: Int, h: Int, old_w: Int, old_h: Int) {
        super.onSizeChanged(w, h, old_w, old_h)
        setBackgroundColor(Color.argb(255, 0, 0, 0))
    }


    override fun onDraw(canvas: Canvas?) {
        canvas?.drawPath(doodlePath, doodlePaint)
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onTouchEvent(event: MotionEvent): Boolean {
        val x = event.x
        val y = event.y
        when (event.action) {
            MotionEvent.ACTION_DOWN -> doodlePath.moveTo(x, y)
            MotionEvent.ACTION_MOVE -> doodlePath.lineTo(x, y)
            else -> return super.onTouchEvent(event)
        }
        invalidate()
        return true
    }

    fun clear() {
        doodlePath.reset()
        invalidate()
    }

    private fun resizeBitmap(src: Bitmap, width: Int, height: Int): Bitmap {
        val matrix = Matrix()
        val src_width = src.width
        val src_height = src.height

        val scale_w = (width.toFloat() / src_width)
        val scale_h = (height.toFloat() / src_height)

        matrix.postScale(scale_w, scale_h)

        val output = Bitmap.createBitmap(src, 0, 0, src_width, src_height, matrix,
                false)

        src.recycle()
        return output
    }

    fun done(viewToPresenter: MVP.ViewToPresenter, demo_params: IntArray) {

        // get the bitmap
        val snapShot = getDrawingCache(true)

        val final_doodle = resizeBitmap(snapShot,28,28)
//        // scale the snapshot to 28x28
//        val scaled_snapShot = resizeBitmap(snapShot, 28, 280)
//
        val doodleHeight = final_doodle.height
        val doodleWidth = final_doodle.width
//
//        // get the pixel values in array
        val pixels = getPixelValues(doodleHeight, doodleWidth, final_doodle)

        System.out.println("CustomCanvas")
        for (i in 0 until doodleHeight * doodleWidth) {
            if (i % doodleWidth == 0)
                System.out.print("\n")
            System.out.print(pixels[i])
        }

        // pass this array to the presenter
        viewToPresenter.donePressed(pixels)
        destroyDrawingCache()
    }

    private fun getPixelValues(doodleHeight: Int, doodleWidth: Int, doodle: Bitmap): IntArray {
        val pixels = IntArray(doodleHeight * doodleWidth)

        doodle.getPixels(pixels, 0, doodleWidth, 0, 0, doodleWidth, doodleHeight)

        for (i in 0 until doodleHeight)
            for (j in 0 until doodleWidth)
                pixels[i * doodleWidth + j] = getGrayScalePixel(doodle, i, j)

        val t_pixels = IntArray(doodleHeight * doodleWidth)

        for (i in 0 until doodleHeight)
            for (j in 0 until doodleWidth)
                t_pixels[i * doodleWidth + j] = pixels[j * doodleWidth + i]

        return t_pixels

    }

    private fun getGrayScalePixel(doodle: Bitmap, i: Int, j: Int): Int {
        val pixel = doodle.getPixel(i, j)

        val r: Int = ((pixel shr 16) and 0xff)
        val g: Int = ((pixel shr 8) and 0xff)
        val b = (pixel and 0xff)
        return if ((r + g + b) == 0) 0 else 1
    }

}