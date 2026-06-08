import 'package:flutter/material.dart';
import 'screens/home_screen.dart';
import 'services/api_service.dart';

void main() {
  runApp(const JarvisApp());
}

class JarvisApp extends StatelessWidget {
  const JarvisApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Jarvis',
      debugShowCheckedModeBanner: false,
      theme: _buildDarkTheme(),
      home: HomeScreen(apiService: ApiService()),
    );
  }

  ThemeData _buildDarkTheme() {
    const bgPrimary = Color(0xFF0d1117);
    const bgSecondary = Color(0xFF161b22);
    const bgTertiary = Color(0xFF21262d);
    const border = Color(0xFF30363d);
    const textPrimary = Color(0xFFe6edf3);
    const textSecondary = Color(0xFF8b949e);
    const accent = Color(0xFF58a6ff);
    const green = Color(0xFF3fb950);
    const red = Color(0xFFf85149);
    const yellow = Color(0xFFd29922);

    return ThemeData(
      brightness: Brightness.dark,
      scaffoldBackgroundColor: bgPrimary,
      colorScheme: const ColorScheme.dark(
        primary: accent,
        secondary: accent,
        surface: bgSecondary,
        error: red,
      ),
      appBarTheme: const AppBarTheme(
        backgroundColor: bgSecondary,
        foregroundColor: textPrimary,
        elevation: 0,
      ),
      cardTheme: CardThemeData(
        color: bgSecondary,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(8),
          side: const BorderSide(color: border),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: bgTertiary,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(6),
          borderSide: const BorderSide(color: border),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(6),
          borderSide: const BorderSide(color: border),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(6),
          borderSide: const BorderSide(color: accent),
        ),
        labelStyle: const TextStyle(color: textSecondary),
        hintStyle: const TextStyle(color: textSecondary),
      ),
      floatingActionButtonTheme: const FloatingActionButtonThemeData(
        backgroundColor: accent,
        foregroundColor: Colors.white,
      ),
      snackBarTheme: SnackBarThemeData(
        backgroundColor: bgTertiary,
        contentTextStyle: const TextStyle(color: textPrimary),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(6)),
      ),
      dividerTheme: const DividerThemeData(color: border),
      extensions: [
        _JarvisColors(
          green: green,
          red: red,
          yellow: yellow,
          border: border,
          bgTertiary: bgTertiary,
          textSecondary: textSecondary,
        ),
      ],
    );
  }
}

// Custom theme extension for Jarvis-specific colors
class _JarvisColors extends ThemeExtension<_JarvisColors> {
  final Color green;
  final Color red;
  final Color yellow;
  final Color border;
  final Color bgTertiary;
  final Color textSecondary;

  const _JarvisColors({
    required this.green,
    required this.red,
    required this.yellow,
    required this.border,
    required this.bgTertiary,
    required this.textSecondary,
  });

  @override
  ThemeExtension<_JarvisColors> copyWith({
    Color? green,
    Color? red,
    Color? yellow,
    Color? border,
    Color? bgTertiary,
    Color? textSecondary,
  }) {
    return _JarvisColors(
      green: green ?? this.green,
      red: red ?? this.red,
      yellow: yellow ?? this.yellow,
      border: border ?? this.border,
      bgTertiary: bgTertiary ?? this.bgTertiary,
      textSecondary: textSecondary ?? this.textSecondary,
    );
  }

  @override
  ThemeExtension<_JarvisColors> lerp(
      covariant ThemeExtension<_JarvisColors>? other, double t) {
    if (other is! _JarvisColors) return this;
    return _JarvisColors(
      green: Color.lerp(green, other.green, t)!,
      red: Color.lerp(red, other.red, t)!,
      yellow: Color.lerp(yellow, other.yellow, t)!,
      border: Color.lerp(border, other.border, t)!,
      bgTertiary: Color.lerp(bgTertiary, other.bgTertiary, t)!,
      textSecondary: Color.lerp(textSecondary, other.textSecondary, t)!,
    );
  }
}

// Helper extension to access Jarvis colors from BuildContext
extension JarvisTheme on BuildContext {
  _JarvisColors get jarvis => Theme.of(this).extension<_JarvisColors>()!;
}
