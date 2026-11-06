package br.edu.app.librasbridge.util

import android.Manifest

object PermissionUtils {
    val requiredPermissions = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.RECORD_AUDIO
    )
}
