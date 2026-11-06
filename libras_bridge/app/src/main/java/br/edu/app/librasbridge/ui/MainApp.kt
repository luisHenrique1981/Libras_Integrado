package br.edu.app.librasbridge.ui

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import br.edu.app.librasbridge.ui.screens.CameraTranslateScreen
import br.edu.app.librasbridge.ui.screens.HomeScreen
import br.edu.app.librasbridge.ui.screens.VoiceToLibrasScreen
import br.edu.app.librasbridge.viewmodel.FraseViewModel
import br.edu.app.librasbridge.viewmodel.MainViewModel

@Composable
fun MainApp() {
    val navController = rememberNavController()
    val cameraViewModel: MainViewModel = viewModel()
    val fraseViewModel: FraseViewModel = viewModel()

    MaterialTheme {
        Surface(modifier = Modifier) {
            NavHost(
                navController = navController,
                startDestination = "home"
            ) {
                composable("home") {
                    HomeScreen(
                        onCameraMode = { navController.navigate("camera") },
                        onVoiceMode = { navController.navigate("voice") }
                    )
                }

                composable("camera") {
                    CameraTranslateScreen(viewModel = cameraViewModel)
                }

                composable("voice") {
                    VoiceToLibrasScreen(viewModel = fraseViewModel)
                }
            }
        }
    }
}