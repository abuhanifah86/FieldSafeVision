import React, { useEffect, useState } from "react";
import { apiFetch, getDefaultApiBase } from "./api";
import { useStoredState } from "./useStoredState";

type Entry = {
  id: number;
  user_id: number;
  user_email: string;
  narrative: string;
  caption?: string | null;
  image_path?: string | null;
  image_mime?: string | null;
  image_size?: number | null;
  created_at: string;
  site?: string | null;
  location?: string | null;
  tags?: string | null;
  severity?: string | null;
  status: string;
  assignee?: string | null;
  notes?: string | null;
  closure_notes?: string | null;
  updated_at: string;
};

type AdminUser = { id: number; email: string; created_at: string };

type Summary = { period: string; count: number };

const AdminPage: React.FC = () => {
  const defaultApi = getDefaultApiBase();
  const [apiBase, setApiBase] = useStoredState<string>("admin.apiBase", defaultApi);
  const [email, setEmail] = useStoredState<string>("admin.email", "");
  const [password, setPassword] = useState("");
  const [token, setToken] = useStoredState<string | null>("admin.token", null);
  const [status, setStatus] = useState("Idle");
  const [entries, setEntries] = useState<Entry[]>([]);
  const [summaries, setSummaries] = useState<Summary[]>([]);
  const [drafts, setDrafts] = useState<Record<number, Partial<Entry>>>({});
  const [admins, setAdmins] = useState<AdminUser[]>([]);
  const [newAdminEmail, setNewAdminEmail] = useState("");
  const [newAdminPassword, setNewAdminPassword] = useState("");

  useEffect(() => {
    if (token) {
      loadEntries();
      loadSummary("daily");
      loadAdmins();
    }
  }, [token]);

  const login = async () => {
    setStatus("Logging in…");
    try {
      const json = await apiFetch<{ token: string }>(`${apiBase}/api/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      setToken(json.token);
      setStatus("Admin authenticated");
      await loadEntries(json.token);
      await loadSummary("daily", json.token);
      await loadAdmins(json.token);
    } catch (err) {
      console.error(err);
      setStatus(
        err instanceof Error
          ? `Login failed: ${err.message}`
          : "Login failed. Check backend URL/credentials.",
      );
    }
  };

  const loadEntries = async (overrideToken?: string) => {
    const useToken = overrideToken ?? token;
    if (!useToken) return;
    setStatus("Loading entries…");
    try {
      const data = await apiFetch<Entry[]>(`${apiBase}/api/entries`, {
        headers: { Authorization: `Bearer ${useToken}` },
      });
      setEntries(data);
      setStatus(`Loaded ${data.length} entries`);
    } catch (err) {
      console.error(err);
      setStatus("Failed to load entries.");
    }
  };

  const loadSummary = async (period: "daily" | "weekly" | "monthly", overrideToken?: string) => {
    const useToken = overrideToken ?? token;
    if (!useToken) return;
    try {
      const data = await apiFetch<Summary[]>(`${apiBase}/api/summary?period=${period}`, {
        headers: { Authorization: `Bearer ${useToken}` },
      });
      setSummaries(data);
      setStatus(`Summary (${period}) loaded`);
    } catch (err) {
      console.error(err);
      setStatus("Failed to load summary.");
    }
  };

  const loadAdmins = async (overrideToken?: string) => {
    const useToken = overrideToken ?? token;
    if (!useToken) return;
    try {
      const data = await apiFetch<AdminUser[]>(`${apiBase}/api/admins`, {
        headers: { Authorization: `Bearer ${useToken}` },
      });
      setAdmins(data);
      setStatus(`Loaded ${data.length} admins`);
    } catch (err) {
      console.error(err);
      setStatus("Failed to load admins.");
    }
  };

  const createAdmin = async () => {
    if (!token) return;
    setStatus("Creating admin…");
    try {
      await apiFetch<AdminUser>(`${apiBase}/api/admins`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email: newAdminEmail, password: newAdminPassword }),
      });
      setNewAdminEmail("");
      setNewAdminPassword("");
      await loadAdmins();
      setStatus("Admin created");
    } catch (err) {
      console.error(err);
      setStatus(
        err instanceof Error ? `Create admin failed: ${err.message}` : "Create admin failed.",
      );
    }
  };

  const deleteAdmin = async (id: number, adminEmail: string) => {
    if (!token) return;
    if (adminEmail === email) {
      setStatus("Cannot remove the currently signed-in admin.");
      return;
    }
    setStatus(`Removing admin ${adminEmail}…`);
    try {
      await apiFetch(`${apiBase}/api/admins/${id}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${token}` },
      });
      await loadAdmins();
      setStatus(`Removed admin ${adminEmail}`);
    } catch (err) {
      console.error(err);
      setStatus(
        err instanceof Error ? `Delete admin failed: ${err.message}` : "Delete admin failed.",
      );
    }
  };

  const deleteEntry = async (id: number) => {
    if (!token) return;
    setStatus(`Deleting entry ${id}…`);
    try {
      await apiFetch(`${apiBase}/api/entries/${id}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${token}` },
      });
      await loadEntries();
      setStatus(`Deleted entry ${id}`);
    } catch (err) {
      console.error(err);
      setStatus("Delete failed.");
    }
  };

  const deleteAll = async () => {
    if (!token) return;
    setStatus("Deleting all entries…");
    try {
      await apiFetch(`${apiBase}/api/entries`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${token}` },
      });
      await loadEntries();
      setStatus("Deleted all entries");
    } catch (err) {
      console.error(err);
      setStatus("Delete all failed.");
    }
  };

  const updateEntry = async (id: number) => {
    if (!token) return;
    const draft = drafts[id] ?? {};
    setStatus(`Updating entry ${id}…`);
    try {
      await apiFetch(`${apiBase}/api/entries/${id}`, {
        method: "PATCH",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          status: draft.status,
          assignee: draft.assignee,
          severity: draft.severity,
          tags: draft.tags,
          notes: draft.notes,
          closure_notes: draft.closure_notes,
        }),
      });
      await loadEntries();
      setStatus(`Updated entry ${id}`);
      setDrafts((prev) => {
        const next = { ...prev };
        delete next[id];
        return next;
      });
    } catch (err) {
      console.error(err);
      setStatus("Update failed.");
    }
  };

  const viewImage = async (id: number) => {
    if (!token) return;
    setStatus(`Fetching image ${id}…`);
    try {
      const res = await fetch(`${apiBase}/api/entries/${id}/image`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) {
        throw new Error(await res.text());
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      window.open(url, "_blank", "noopener,noreferrer");
      setStatus(`Opened image ${id}`);
    } catch (err) {
      console.error(err);
      setStatus("Failed to fetch image (auth or network issue).");
    }
  };

  return (
    <>
      <div className="card upload">
        <label htmlFor="adminApi">Backend URL</label>
        <input
          id="adminApi"
          type="text"
          value={apiBase}
          onChange={(e) => setApiBase(e.target.value)}
          placeholder="https://your-backend:8000"
        />
        <strong>Admin login</strong>
        <input
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="admin email"
        />
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="password"
        />
        <div className="buttons">
          <button type="button" onClick={login} disabled={!email || !password}>
            Login
          </button>
          <button type="button" className="secondary" onClick={() => loadEntries()}>
            Refresh entries
          </button>
          <button
            type="button"
            className="secondary"
            onClick={() => {
              setToken(null);
              setEntries([]);
              setSummaries([]);
              setStatus("Logged out");
            }}
          >
            Logout
          </button>
        </div>
        <div className="status">Token: {token ? "present" : "none"}</div>
      </div>

      {token && (
        <>
          <div className="card">
            <strong>Admin management</strong>
            <div className="upload" style={{ gap: 8 }}>
              <div className="buttons">
                <button type="button" className="secondary" onClick={() => loadAdmins()}>
                  Refresh admins
                </button>
              </div>
              <div className="status">Total admins: {admins.length}</div>
              {admins.map((admin) => (
                <div key={admin.id} className="status" style={{ display: "flex", gap: 8, alignItems: "center" }}>
                  <span>
                    {admin.email} • created {new Date(admin.created_at).toLocaleString()}
                  </span>
                  <button
                    className="secondary"
                    onClick={() => deleteAdmin(admin.id, admin.email)}
                    disabled={admin.email === email || admins.length <= 1}
                  >
                    Remove
                  </button>
                </div>
              ))}
              <div className="upload" style={{ gap: 8 }}>
                <input
                  type="email"
                  placeholder="new admin email"
                  value={newAdminEmail}
                  onChange={(e) => setNewAdminEmail(e.target.value)}
                />
                <input
                  type="password"
                  placeholder="password (min 6 chars)"
                  value={newAdminPassword}
                  onChange={(e) => setNewAdminPassword(e.target.value)}
                />
                <button
                  onClick={createAdmin}
                  disabled={!newAdminEmail || !newAdminPassword || newAdminPassword.length < 6}
                >
                  Add admin
                </button>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="buttons">
              <button onClick={() => loadSummary("daily")} className="secondary">
                Daily summary
              </button>
              <button onClick={() => loadSummary("weekly")} className="secondary">
                Weekly summary
              </button>
              <button onClick={() => loadSummary("monthly")} className="secondary">
                Monthly summary
              </button>
              <button onClick={deleteAll} className="secondary">
                Delete all
              </button>
            </div>
            {summaries.length > 0 && (
              <div className="response">
                {summaries.map((s) => (
                  <div key={s.period}>
                    {s.period}: {s.count} entries
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="card">
            <strong>Entries</strong>
            {entries.length === 0 && <div className="status">No entries.</div>}
            <div className="upload" style={{ gap: 8 }}>
              {entries.map((entry) => {
                const draft = drafts[entry.id] ?? {
                  status: entry.status,
                  assignee: entry.assignee ?? "",
                  severity: entry.severity ?? "",
                  tags: entry.tags ?? "",
                  notes: entry.notes ?? "",
                  closure_notes: entry.closure_notes ?? "",
                };
                return (
                  <div key={entry.id} className="card" style={{ padding: 12 }}>
                    <div className="status">
                      #{entry.id} • User {entry.user_id} ({entry.user_email}) •{" "}
                      {new Date(entry.created_at).toLocaleString()}
                    </div>
                    <div className="status">
                      Site: {entry.site || "n/a"} | Location: {entry.location || "n/a"} | Severity:{" "}
                      {entry.severity || "n/a"} | Status: {entry.status} | Assignee:{" "}
                      {entry.assignee || "n/a"}
                    </div>
                    {entry.tags && <div className="status">Tags: {entry.tags}</div>}
                    {entry.notes && <div className="status">Notes: {entry.notes}</div>}
                    {entry.closure_notes && (
                      <div className="status">Closure notes: {entry.closure_notes}</div>
                    )}
                    <div className="response">{entry.narrative}</div>
                    {entry.caption && <div className="status">Caption: {entry.caption}</div>}
                    <div className="buttons">
                      <button className="secondary" onClick={() => viewImage(entry.id)} disabled={!token}>
                        View image
                      </button>
                      <button className="secondary" onClick={() => deleteEntry(entry.id)}>
                        Delete
                      </button>
                    </div>
                    <div className="upload" style={{ gap: 8 }}>
                      <input
                        type="text"
                        placeholder="Assignee"
                        value={draft.assignee ?? ""}
                        onChange={(e) =>
                          setDrafts((prev) => ({
                            ...prev,
                            [entry.id]: { ...draft, assignee: e.target.value },
                          }))
                        }
                      />
                      <select
                        value={draft.status ?? entry.status}
                        onChange={(e) =>
                          setDrafts((prev) => ({
                            ...prev,
                            [entry.id]: { ...draft, status: e.target.value },
                          }))
                        }
                      >
                        <option value="open">Open</option>
                        <option value="under_review">Under review</option>
                        <option value="closed">Closed</option>
                      </select>
                      <select
                        value={draft.severity ?? entry.severity ?? ""}
                        onChange={(e) =>
                          setDrafts((prev) => ({
                            ...prev,
                            [entry.id]: { ...draft, severity: e.target.value },
                          }))
                        }
                      >
                        <option value="">Severity</option>
                        <option value="minor">Minor</option>
                        <option value="major">Major</option>
                        <option value="critical">Critical</option>
                      </select>
                      <input
                        type="text"
                        placeholder="Tags"
                        value={draft.tags ?? entry.tags ?? ""}
                        onChange={(e) =>
                          setDrafts((prev) => ({
                            ...prev,
                            [entry.id]: { ...draft, tags: e.target.value },
                          }))
                        }
                      />
                      <textarea
                        placeholder="Notes"
                        value={draft.notes ?? entry.notes ?? ""}
                        onChange={(e) =>
                          setDrafts((prev) => ({
                            ...prev,
                            [entry.id]: { ...draft, notes: e.target.value },
                          }))
                        }
                        rows={2}
                        style={{ width: "100%", font: "inherit" }}
                      />
                      <textarea
                        placeholder="Closure notes"
                        value={draft.closure_notes ?? entry.closure_notes ?? ""}
                        onChange={(e) =>
                          setDrafts((prev) => ({
                            ...prev,
                            [entry.id]: { ...draft, closure_notes: e.target.value },
                          }))
                        }
                        rows={2}
                        style={{ width: "100%", font: "inherit" }}
                      />
                      <button onClick={() => updateEntry(entry.id)} disabled={!token}>
                        Save updates
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </>
      )}
      <div className="status">{status}</div>
    </>
  );
};

export default AdminPage;
