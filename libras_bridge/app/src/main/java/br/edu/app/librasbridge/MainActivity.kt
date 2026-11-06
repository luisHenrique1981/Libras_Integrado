package br.edu.app.librasbridge

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import br.edu.app.librasbridge.avatar.SignDictionary
import br.edu.app.librasbridge.ui.MainApp

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        SignDictionary.initialize(this)
        setContent {
            MainApp()
        }
    }
}
