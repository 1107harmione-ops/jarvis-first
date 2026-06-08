import 'package:flutter/material.dart';
import '../models/reminder.dart';
import '../services/api_service.dart';
import '../main.dart';

class RemindersScreen extends StatefulWidget {
  final ApiService api;
  const RemindersScreen({super.key, required this.api});

  @override
  State<RemindersScreen> createState() => _RemindersScreenState();
}

class _RemindersScreenState extends State<RemindersScreen> {
  List<Reminder> _reminders = [];
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
      final result = await widget.api.getReminders();
      setState(() { _reminders = result.reminders; _loading = false; });
    } catch (e) {
      setState(() { _error = e.toString(); _loading = false; });
    }
  }

  Future<void> _create() async {
    final titleCtl = TextEditingController();
    DateTime selectedDate = DateTime.now().add(const Duration(hours: 1));

    final result = await showDialog<bool>(
      context: context,
      builder: (ctx) => StatefulBuilder(
        builder: (ctx, setDialogState) => AlertDialog(
          title: const Text('New Reminder'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              TextField(controller: titleCtl, decoration: const InputDecoration(labelText: 'What'), autofocus: true),
              const SizedBox(height: 12),
              ListTile(
                leading: const Icon(Icons.calendar_today),
                title: const Text('When'),
                subtitle: Text('${selectedDate.month}/${selectedDate.day}/${selectedDate.year} ${selectedDate.hour.toString().padLeft(2, '0')}:${selectedDate.minute.toString().padLeft(2, '0')}'),
                onTap: () async {
                  final date = await showDatePicker(context: ctx, initialDate: selectedDate, firstDate: DateTime.now(), lastDate: DateTime.now().add(const Duration(days: 365)));
                  if (date != null) {
                    final time = await showTimePicker(context: ctx, initialTime: TimeOfDay.fromDateTime(selectedDate));
                    if (time != null) {
                      setDialogState(() {
                        selectedDate = DateTime(date.year, date.month, date.day, time.hour, time.minute);
                      });
                    }
                  }
                },
              ),
            ],
          ),
          actions: [
            TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
            FilledButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('Create')),
          ],
        ),
      ),
    );

    if (result == true && titleCtl.text.trim().isNotEmpty) {
      try {
        await widget.api.createReminder(
          title: titleCtl.text.trim(),
          reminderTime: selectedDate.toIso8601String(),
        );
        _load();
      } catch (e) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error: $e')));
        }
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 8, 16, 0),
            child: Row(
              children: [
                Text('Reminders', style: Theme.of(context).textTheme.titleMedium),
                const Spacer(),
                FilledButton.icon(
                  icon: const Icon(Icons.add, size: 18),
                  label: const Text('New Reminder'),
                  onPressed: _create,
                ),
              ],
            ),
          ),
          const SizedBox(height: 8),
          Expanded(
            child: _loading
                ? const Center(child: CircularProgressIndicator())
                : _error != null
                    ? Center(child: Text('Error: $_error'))
                    : _reminders.isEmpty
                        ? Center(
                            child: Column(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Icon(Icons.alarm, size: 48, color: context.jarvis.textSecondary),
                                const SizedBox(height: 12),
                                Text('No reminders yet.', style: TextStyle(color: context.jarvis.textSecondary)),
                              ],
                            ),
                          )
                        : RefreshIndicator(
                            onRefresh: _load,
                            child: ListView.builder(
                              padding: const EdgeInsets.symmetric(horizontal: 16),
                              itemCount: _reminders.length,
                              itemBuilder: (ctx, i) {
                                final r = _reminders[i];
                                return Card(
                                  child: ListTile(
                                    leading: Icon(Icons.alarm, color: r.triggered ? Colors.grey : Theme.of(context).colorScheme.primary),
                                    title: Text(r.title),
                                    subtitle: Column(
                                      crossAxisAlignment: CrossAxisAlignment.start,
                                      children: [
                                        Text(r.formattedTime, style: TextStyle(color: context.jarvis.textSecondary, fontSize: 12)),
                                        Row(
                                          children: [
                                            Container(
                                              padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                                              decoration: BoxDecoration(
                                                color: context.jarvis.bgTertiary,
                                                borderRadius: BorderRadius.circular(10),
                                              ),
                                              child: Text(r.status, style: const TextStyle(fontSize: 10)),
                                            ),
                                            if (r.repeatType != 'none') ...[
                                              const SizedBox(width: 8),
                                              Container(
                                                padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                                                decoration: BoxDecoration(
                                                  color: context.jarvis.bgTertiary,
                                                  borderRadius: BorderRadius.circular(10),
                                                ),
                                                child: Text('repeats: ${r.repeatType}', style: const TextStyle(fontSize: 10)),
                                              ),
                                            ],
                                          ],
                                        ),
                                      ],
                                    ),
                                    trailing: IconButton(
                                      icon: Icon(Icons.delete_outline, color: context.jarvis.red),
                                      onPressed: () async {
                                        try {
                                          await widget.api.deleteReminder(r.id!);
                                          _load();
                                        } catch (e) {
                                          if (mounted) {
                                            ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('$e')));
                                          }
                                        }
                                      },
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
