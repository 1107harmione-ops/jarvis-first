package com.jarvis.voice.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import kotlin.math.abs

/**
 * Custom view that visualizes audio level as animated bars.
 *
 * Displays a waveform-like visualization that responds to
 * the current audio input level in real-time.
 */
class AudioLevelView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0,
) : View(context, attrs, defStyleAttr) {

    /** Number of bars in the visualization. */
    private val barCount = 40

    /** Current audio level (0.0–1.0). */
    @Volatile
    private var currentLevel: Float = 0f

    /** Target level for smooth animation. */
    private var targetLevel: Float = 0f

    /** Paint for the bars. */
    private val barPaint = Paint().apply {
        color = Color.parseColor("#1A73E8")
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    /** Paint for active (speaking) bars. */
    private val activePaint = Paint().apply {
        color = Color.parseColor("#4CAF50")
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    /** Paint for background bars. */
    private val backgroundPaint = Paint().apply {
        color = Color.parseColor("#333333")
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    // ── Public API ──────────────────────────────────────────────

    /**
     * Set the current audio level. Smoothly animates to the target.
     */
    fun setLevel(level: Float) {
        targetLevel = level.coerceIn(0f, 1f)
        invalidate()
    }

    // ── Drawing ─────────────────────────────────────────────────

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        // Smoothly interpolate current level toward target
        currentLevel += (targetLevel - currentLevel) * 0.3f
        if (abs(currentLevel - targetLevel) < 0.01f) {
            currentLevel = targetLevel
        }

        val width = width.toFloat()
        val height = height.toFloat()
        val barWidth = width / (barCount * 1.5f)
        val spacing = barWidth * 0.5f

        for (i in 0 until barCount) {
            // Calculate bar height based on position and current level
            val positionFactor = when {
                i < barCount / 2 -> i.toFloat() / (barCount / 2)
                else -> (barCount - i).toFloat() / (barCount / 2)
            }
            val barHeight = (height * 0.8f) * positionFactor * currentLevel
            val barHeightClamped = maxOf(barHeight, 2f)

            val x = i * (barWidth + spacing) + spacing
            val y = (height - barHeightClamped) / 2f

            // Color based on level
            val paint = when {
                currentLevel > 0.6f -> activePaint
                currentLevel > 0.2f -> barPaint
                else -> backgroundPaint
            }

            canvas.drawRoundRect(
                x, y,
                x + barWidth, y + barHeightClamped,
                2f, 2f,
                paint,
            )
        }

        // Continue animation if not at target
        if (abs(currentLevel - targetLevel) > 0.01f) {
            invalidate()
        }
    }
}
