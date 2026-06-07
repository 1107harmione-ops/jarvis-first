# Android Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a basic Android application skeleton in Kotlin that builds successfully.

**Architecture:** Single Activity application using standard Android project structure with Gradle build system. The app contains a MainActivity that sets the content view to a layout file.

**Tech Stack:** Kotlin, Android Gradle Plugin 7.4.2, AndroidX Core, AppCompat, Material Components, ConstraintLayout.

---
### Task 1: Project Settings and Build Configuration

**Files:**
- Create: `settings.gradle`
- Create: `build.gradle`

- [ ] **Step 1: Write settings.gradle**

```gradle
include ':app'
```

- [ ] **Step 2: Write project-level build.gradle**

```gradle
buildscript {
    ext.kotlin_version = '1.8.0'
    repositories {
        google()
        mavenCentral()
    }
    dependencies {
        classpath 'com.android.tools.build:gradle:7.4.2'
        classpath "org.jetbrains.kotlin:kotlin-gradle-plugin:$kotlin_version"
    }
}

allprojects {
    repositories {
        google()
        mavenCentral()
    }
}
```

- [ ] **Step 3: Verify files are created correctly**

Run: `cat settings.gradle build.gradle`
Expected: Files contain the above content

- [ ] **Step 4: Commit initial project configuration**

```bash
git add settings.gradle build.gradle
git commit -m "chore: add project settings and build configuration"
```

### Task 2: Android App Module Configuration

**Files:**
- Create: `app/build.gradle`
- Create: `app/src/main/AndroidManifest.xml`
- Create: `app/src/main/java/com/example/myapp/MainActivity.kt`

- [ ] **Step 1: Write app/build.gradle**

```gradle
plugins {
    id 'com.android.application'
    id 'org.jetbrains.kotlin.android'
}

android {
    compileSdk 34

    defaultConfig {
        applicationId "com.example.myapp"
        minSdk 21
        targetSdk 34
        versionCode 1
        versionName "1.0"

        testInstrumentationRunner "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            minifyEnabled false
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
        }
    }
    compileOptions {
        sourceCompatibility JavaVersion.VERSION_17
        targetCompatibility JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    implementation "androidx.core:core-ktx:1.12.0"
    implementation "androidx.appcompat:appcompat:1.6.1"
    implementation "com.google.android.material:material:1.8.0"
    implementation "androidx.constraintlayout:constraintlayout:2.1.4"
    testImplementation "junit:junit:4.13.2"
    androidTestImplementation "androidx.test.ext:junit:1.1.5"
    androidTestImplementation "androidx.test.espresso:espresso-core:3.5.1"
}
```

- [ ] **Step 2: Write AndroidManifest.xml**

```xml
<?xml version="1.0" encoding="utf-8"?>
<manifest package="com.example.myapp">

    <application
        android:allowBackup="true"
        android:label="@string/app_name"
        android:icon="@mipmap/ic_launcher"
        android:roundIcon="@mipmap/ic_launcher_round"
        android:supportsRtl="true"
        android:theme="@style/Theme.MyApp">
        <activity android:name=".MainActivity"
            android:exported="true">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />

                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
    </application>

</manifest>
```

- [ ] **Step 3: Write MainActivity.kt**

```kotlin
package com.example.myapp

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
    }
}
```

- [ ] **Step 4: Verify module files are created**

Run: `find app -type f -not -path "*/\.*" | sort`
Expected: List of created files

- [ ] **Step 5: Commit app module configuration**

```bash
git add app/
git commit -m "feat: add Android app module with MainActivity"
```

### Task 3: Create Main Layout Resource

**Files:**
- Create: `app/src/main/res/layout/activity_main.xml`
- Create: `app/src/main/res/values/strings.xml`
- Create: `app/src/main/res/values/themes.xml`

- [ ] **Step 1: Write activity_main.xml**

```xml
<?xml version="1.0" encoding="utf-8"?>
<androidx.constraintlayout.widget.ConstraintLayout xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    xmlns:tools="http://schemas.android.com/tools"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    tools:context=".MainActivity">

    <TextView
        android:id="@+id/textView"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="Hello World!"
        app:layout_constraintBottom_toBottomOf="parent"
        android:layout_marginBottom="16dp"
        app:layout_constraintLeft_toLeftOf="parent"
        android:layout_marginLeft="16dp"
        app:layout_constraintRight_toRightOf="parent"
        android:layout_marginRight="16dp"
        app:layout_constraintTop_toTopOf="parent"
        android:layout_marginTop="16dp" />

</androidx.constraintlayout.widget.ConstraintLayout>
```

- [ ] **Step 2: Write strings.xml**

```xml
<resources>
    <string name="app_name">MyApp</string>
</resources>
```

- [ ] **Step 3: Write themes.xml**

```xml
<resources xmlns:tools="http://schemas.android.com/tools">
    <!-- Base application theme. -->
    <style name="Theme.MyApp" parent="Theme.MaterialComponents.DayNight.NoActionBar">
        <!-- Primary brand color. -->
        <item name="colorPrimary">@color/purple_500</item>
        <item name="colorPrimaryVariant">@color/purple_700</item>
        <item name="colorOnPrimary">@color/white</item>
        <!-- Secondary brand color. -->
        <item name="colorSecondary">@color/teal_200</item>
        <item name="colorSecondaryVariant">@color/teal_700</item>
        <item name="colorOnSecondary">@color/black</item>
        <!-- Status bar color. -->
        <item name="android:statusBarColor" tools:targetApi="l">?attr/colorPrimaryVariant</item>
        <!-- Customize your theme here. -->
    </style>
</resources>
```

- [ ] **Step 4: Verify layout resources are created**

Run: `find app/src/main/res -type f -not -path "*/\.*" | sort`
Expected: List of created resource files

- [ ] **Step 5: Commit layout resources**

```bash
git add app/src/main/res/
git commit -m "feat: add main layout and resources"
```

### Task 4: Write Unit Test for MainActivity

**Files:**
- Create: `app/src/test/java/com/example/myapp/MainActivityTest.kt`

- [ ] **Step 1: Write MainActivityTest.kt**

```kotlin
package com.example.myapp

import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Instrumented test, which will execute on an Android device.
 *
 * @see androidx.test.espresso.Espresso
 */
@RunWith(AndroidJUnit4::class)
class MainActivityTest {

    @Test
    fun useAppContext() {
        // Context of the app under test.
        // Not needed for this simple test, but we can check that the activity launches
        ActivityScenario.launch(MainActivity::class.java)
    }
}
```

- [ ] **Step 2: Verify test file is created**

Run: `ls app/src/test/java/com/example/myapp/`
Expected: MainActivityTest.kt

- [ ] **Step 3: Commit test file**

```bash
git add app/src/test/java/com/example/myapp/MainActivityTest.kt
git commit -m "feat: add unit test for MainActivity"
```

### Task 5: Build and Test the Application

**Files:**
- None (verification steps)

- [ ] **Step 1: Build the debug APK**

Run: `./gradlew assembleDebug`
Expected: BUILD SUCCESSFUL

- [ ] **Step 2: Run unit tests**

Run: `./gradlew testDebugUnitTest`
Expected: Tests pass

- [ ] **Step 3: Run connected instrumented tests (requires emulator or device)**

Note: This step requires an emulator or device. We'll skip automatic execution but note the command.
Run: `./gradlew connectedAndroidTest`
Expected: Tests pass (if emulator/device is available)

- [ ] **Step 4: Commit build and test results**

```bash
git commit --allow-empty -m "chore: verify build and test success"
```