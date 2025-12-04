import React from "react";
import { BrowserRouter, Navigate, Route, Routes, Link } from "react-router-dom";
import ObserverPage from "./ObserverPage";
import AdminPage from "./AdminPage";

function App() {
  return (
    <BrowserRouter>
      <div className="app">
        <div className="heading">
          <div className="brand-wrap">
            <span className="brand-icon" aria-hidden="true">
              🪖
            </span>
            <h1 className="brand">FieldSafe Vision</h1>
          </div>
          <nav className="buttons">
            <Link to="/observer" className="secondary">
              Observer
            </Link>
            <Link to="/admin" className="secondary">
              Administrator
            </Link>
          </nav>
        </div>
        <Routes>
          <Route path="/observer" element={<ObserverPage />} />
          <Route path="/admin" element={<AdminPage />} />
          <Route path="*" element={<Navigate to="/observer" replace />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;
