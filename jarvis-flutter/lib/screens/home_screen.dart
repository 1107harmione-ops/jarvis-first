import 'package:flutter/material.dart';
import '../services/api_service.dart';
import 'dashboard_screen.dart';
import 'tasks_screen.dart';
import 'notes_screen.dart';
import 'reminders_screen.dart';
import 'memory_screen.dart';
import 'search_screen.dart';

class HomeScreen extends StatefulWidget {
  final ApiService apiService;
  const HomeScreen({super.key, required this.apiService});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _currentIndex = 0;

  late final List<Widget> _screens;
  late final List<String> _titles;

  @override
  void initState() {
    super.initState();
    _screens = [
      DashboardScreen(api: widget.apiService),
      TasksScreen(api: widget.apiService),
      NotesScreen(api: widget.apiService),
      RemindersScreen(api: widget.apiService),
      MemoryScreen(api: widget.apiService),
      SearchScreen(api: widget.apiService),
    ];
    _titles = [
      'Dashboard',
      'Tasks',
      'Notes',
      'Reminders',
      'Memory',
      'Search',
    ];
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Row(
          children: [
            Container(
              width: 28,
              height: 28,
              decoration: BoxDecoration(
                color: Theme.of(context).colorScheme.primary,
                borderRadius: BorderRadius.circular(6),
              ),
              child: const Center(
                child: Text('J', style: TextStyle(fontWeight: FontWeight.bold, color: Colors.white, fontSize: 16)),
              ),
            ),
            const SizedBox(width: 10),
            const Text('Jarvis'),
          ],
        ),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 12),
            child: Row(
              children: [
                Container(
                  width: 8,
                  height: 8,
                  decoration: const BoxDecoration(
                    color: Color(0xFF3fb950),
                    shape: BoxShape.circle,
                  ),
                ),
                const SizedBox(width: 6),
                Text(
                  'Connected',
                  style: TextStyle(color: Theme.of(context).colorScheme.onSurface.withOpacity(0.6), fontSize: 12),
                ),
              ],
            ),
          ),
        ],
      ),
      drawer: _buildDrawer(context),
      body: IndexedStack(
        index: _currentIndex,
        children: _screens,
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _currentIndex,
        onDestinationSelected: (i) => setState(() => _currentIndex = i),
        destinations: const [
          NavigationDestination(icon: Icon(Icons.dashboard_outlined), selectedIcon: Icon(Icons.dashboard), label: 'Dashboard'),
          NavigationDestination(icon: Icon(Icons.checklist_outlined), selectedIcon: Icon(Icons.checklist), label: 'Tasks'),
          NavigationDestination(icon: Icon(Icons.note_outlined), selectedIcon: Icon(Icons.note), label: 'Notes'),
          NavigationDestination(icon: Icon(Icons.alarm_outlined), selectedIcon: Icon(Icons.alarm), label: 'Reminders'),
          NavigationDestination(icon: Icon(Icons.psychology_outlined), selectedIcon: Icon(Icons.psychology), label: 'Memory'),
          NavigationDestination(icon: Icon(Icons.search_outlined), selectedIcon: Icon(Icons.search), label: 'Search'),
        ],
      ),
    );
  }

  Widget _buildDrawer(BuildContext context) {
    final items = [
      ('Dashboard', Icons.dashboard_outlined, 0),
      ('Tasks', Icons.checklist_outlined, 1),
      ('Notes', Icons.note_outlined, 2),
      ('Reminders', Icons.alarm_outlined, 3),
      ('Memory', Icons.psychology_outlined, 4),
      ('Search', Icons.search_outlined, 5),
    ];

    return Drawer(
      child: ListView(
        padding: EdgeInsets.zero,
        children: [
          DrawerHeader(
            decoration: const BoxDecoration(color: Color(0xFF161b22)),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisAlignment: MainAxisAlignment.end,
              children: [
                Container(
                  width: 40,
                  height: 40,
                  decoration: BoxDecoration(
                    color: Theme.of(context).colorScheme.primary,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: const Center(
                    child: Text('J', style: TextStyle(fontWeight: FontWeight.bold, color: Colors.white, fontSize: 22)),
                  ),
                ),
                const SizedBox(height: 12),
                const Text('Jarvis', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600)),
                Text('Voice Productivity Assistant', style: TextStyle(fontSize: 12, color: Colors.grey[400])),
              ],
            ),
          ),
          ...items.map((item) => ListTile(
                leading: Icon(_currentIndex == item.$3 ? item.$2 : item.$2),
                title: Text(item.$1),
                selected: _currentIndex == item.$3,
                onTap: () {
                  setState(() => _currentIndex = item.$3);
                  Navigator.pop(context);
                },
              )),
          const Divider(),
          ListTile(
            leading: const Icon(Icons.settings_outlined),
            title: const Text('Settings'),
            onTap: () => Navigator.pop(context),
          ),
        ],
      ),
    );
  }
}
