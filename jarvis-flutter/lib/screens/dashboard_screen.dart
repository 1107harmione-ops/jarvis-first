import 'package:flutter/material.dart';
import '../services/api_service.dart';
import '../main.dart';

class DashboardScreen extends StatefulWidget {
  final ApiService api;
  const DashboardScreen({super.key, required this.api});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  int _taskCount = 0;
  int _pendingCount = 0;
  int _noteCount = 0;
  int _memoryCount = 0;
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() { _loading = true; _error = null; });
    try {
      final results = await Future.wait([
        widget.api.getTasks(),
        widget.api.getNotes(limit: 3),
        widget.api.getMemories(),
      ]);
      final tasks = results[0] as TaskListResponse;
      final notes = results[1] as NoteListResponse;
      final mem = results[2] as MemoryListResponse;

      setState(() {
        _taskCount = tasks.total;
        _pendingCount = tasks.tasks.where((t) => t.status != 'completed').length;
        _noteCount = notes.total;
        _memoryCount = mem.total;
        _loading = false;
      });
    } catch (e) {
      setState(() { _error = e.toString(); _loading = false; });
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return const Center(child: CircularProgressIndicator());
    }
    if (_error != null) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.cloud_off, size: 48, color: Colors.grey),
            const SizedBox(height: 16),
            Text('Could not connect', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            TextButton(onPressed: _load, child: const Text('Retry')),
          ],
        ),
      );
    }

    return RefreshIndicator(
      onRefresh: _load,
      child: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // Stats Grid
          GridView.count(
            crossAxisCount: 2,
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            mainAxisSpacing: 12,
            crossAxisSpacing: 12,
            childAspectRatio: 1.6,
            children: [
              _StatCard(value: '$_taskCount', label: 'Tasks', icon: Icons.checklist),
              _StatCard(value: '$_pendingCount', label: 'Pending', icon: Icons.hourglass_empty),
              _StatCard(value: '$_noteCount', label: 'Notes', icon: Icons.note),
              _StatCard(value: '$_memoryCount', label: 'Memories', icon: Icons.psychology),
            ],
          ),
          const SizedBox(height: 24),
          Text('Recent Activity', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 12),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(32),
              child: Column(
                children: [
                  Icon(Icons.inbox_outlined, size: 48, color: context.jarvis.textSecondary),
                  const SizedBox(height: 12),
                  Text('No recent activity.', style: TextStyle(color: context.jarvis.textSecondary)),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _StatCard extends StatelessWidget {
  final String value;
  final String label;
  final IconData icon;
  const _StatCard({required this.value, required this.label, required this.icon});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(icon, color: Theme.of(context).colorScheme.primary, size: 20),
          const SizedBox(height: 8),
          Text(value, style: const TextStyle(fontSize: 28, fontWeight: FontWeight.bold)),
          const SizedBox(height: 4),
          Text(label, style: TextStyle(fontSize: 12, color: context.jarvis.textSecondary)),
        ],
      ),
    );
  }
}
