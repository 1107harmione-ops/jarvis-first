import 'package:flutter/material.dart';
import '../models/task.dart';
import '../services/api_service.dart';
import '../main.dart';

class TasksScreen extends StatefulWidget {
  final ApiService api;
  const TasksScreen({super.key, required this.api});

  @override
  State<TasksScreen> createState() => _TasksScreenState();
}

class _TasksScreenState extends State<TasksScreen> {
  List<Task> _tasks = [];
  bool _loading = true;
  String? _error;
  String _statusFilter = '';

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() { _loading = true; _error = null; });
    try {
      final result = await widget.api.getTasks(status: _statusFilter.isEmpty ? null : _statusFilter);
      setState(() { _tasks = result.tasks; _loading = false; });
    } catch (e) {
      setState(() { _error = e.toString(); _loading = false; });
    }
  }

  Future<void> _createOrUpdate({Task? task}) async {
    final isEdit = task != null;
    final titleCtl = TextEditingController(text: task?.title ?? '');
    final descCtl = TextEditingController(text: task?.description ?? '');
    String priority = task?.priority ?? 'medium';

    final result = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(isEdit ? 'Edit Task' : 'New Task'),
        content: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              TextField(controller: titleCtl, decoration: const InputDecoration(labelText: 'Title'), autofocus: true),
              const SizedBox(height: 12),
              TextField(controller: descCtl, decoration: const InputDecoration(labelText: 'Description'), maxLines: 3),
              const SizedBox(height: 12),
              DropdownButtonFormField<String>(
                value: priority,
                decoration: const InputDecoration(labelText: 'Priority'),
                items: ['low', 'medium', 'high'].map((p) => DropdownMenuItem(value: p, child: Text(p))).toList(),
                onChanged: (v) => priority = v ?? 'medium',
              ),
            ],
          ),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
          FilledButton(onPressed: () => Navigator.pop(ctx, true), child: Text(isEdit ? 'Update' : 'Create')),
        ],
      ),
    );

    if (result == true && titleCtl.text.trim().isNotEmpty) {
      try {
        if (isEdit) {
          await widget.api.updateTask(task!.id!, title: titleCtl.text.trim(), description: descCtl.text.trim(), priority: priority);
        } else {
          await widget.api.createTask(title: titleCtl.text.trim(), description: descCtl.text.trim(), priority: priority);
        }
        _load();
      } catch (e) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error: $e')));
        }
      }
    }
  }

  Color _priorityColor(String p) {
    switch (p) {
      case 'high': return context.jarvis.red;
      case 'medium': return context.jarvis.yellow;
      default: return context.jarvis.green;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        children: [
          // Filter
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 8, 16, 0),
            child: Row(
              children: [
                DropdownButton<String>(
                  value: _statusFilter,
                  hint: const Text('All'),
                  items: const [
                    DropdownMenuItem(value: '', child: Text('All')),
                    DropdownMenuItem(value: 'pending', child: Text('Pending')),
                    DropdownMenuItem(value: 'completed', child: Text('Completed')),
                  ],
                  onChanged: (v) {
                    setState(() => _statusFilter = v ?? '');
                    _load();
                  },
                ),
                const Spacer(),
                FilledButton.icon(
                  icon: const Icon(Icons.add, size: 18),
                  label: const Text('New Task'),
                  onPressed: () => _createOrUpdate(),
                ),
              ],
            ),
          ),
          const SizedBox(height: 8),
          // List
          Expanded(
            child: _loading
                ? const Center(child: CircularProgressIndicator())
                : _error != null
                    ? Center(child: Text('Error: $_error'))
                    : _tasks.isEmpty
                        ? Center(
                            child: Column(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Icon(Icons.checklist, size: 48, color: context.jarvis.textSecondary),
                                const SizedBox(height: 12),
                                Text('No tasks yet.', style: TextStyle(color: context.jarvis.textSecondary)),
                              ],
                            ),
                          )
                        : RefreshIndicator(
                            onRefresh: _load,
                            child: ListView.builder(
                              padding: const EdgeInsets.symmetric(horizontal: 16),
                              itemCount: _tasks.length,
                              itemBuilder: (ctx, i) {
                                final t = _tasks[i];
                                final isDone = t.status == 'completed';
                                return Card(
                                  child: ListTile(
                                    leading: IconButton(
                                      icon: Icon(
                                        isDone ? Icons.check_circle : Icons.radio_button_unchecked,
                                        color: isDone ? context.jarvis.green : context.jarvis.textSecondary,
                                      ),
                                      onPressed: isDone ? null : () async {
                                        try {
                                          await widget.api.completeTask(t.id!);
                                          _load();
                                        } catch (e) {
                                          if (mounted) {
                                            ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('$e')));
                                          }
                                        }
                                      },
                                    ),
                                    title: Text(t.title, style: isDone ? const TextStyle(decoration: TextDecoration.lineThrough, color: Colors.grey) : null),
                                    subtitle: Column(
                                      crossAxisAlignment: CrossAxisAlignment.start,
                                      children: [
                                        if (t.description != null && t.description!.isNotEmpty)
                                          Text(t.description!, maxLines: 1, overflow: TextOverflow.ellipsis,
                                            style: TextStyle(color: context.jarvis.textSecondary, fontSize: 12)),
                                        const SizedBox(height: 4),
                                        Row(
                                          children: [
                                            Container(
                                              padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                                              decoration: BoxDecoration(
                                                color: _priorityColor(t.priority).withOpacity(0.2),
                                                borderRadius: BorderRadius.circular(10),
                                              ),
                                              child: Text(t.priority, style: TextStyle(fontSize: 10, color: _priorityColor(t.priority))),
                                            ),
                                            const SizedBox(width: 8),
                                            Container(
                                              padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                                              decoration: BoxDecoration(
                                                color: context.jarvis.bgTertiary,
                                                borderRadius: BorderRadius.circular(10),
                                              ),
                                              child: Text(t.status, style: TextStyle(fontSize: 10, color: context.jarvis.textSecondary)),
                                            ),
                                          ],
                                        ),
                                      ],
                                    ),
                                    trailing: PopupMenuButton(
                                      itemBuilder: (_) => [
                                        PopupMenuItem(child: const Text('Edit'), onTap: () => _createOrUpdate(task: t)),
                                        PopupMenuItem(child: Text('Delete', style: TextStyle(color: context.jarvis.red)), onTap: () async {
                                          try {
                                            await widget.api.deleteTask(t.id!);
                                            _load();
                                          } catch (e) {
                                            if (mounted) {
                                              ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('$e')));
                                            }
                                          }
                                        }),
                                      ],
                                    ),
                                  ),
                                );
                              },
                            ),
                          ),
          ),
        ],
      ),
    );
  }
}
