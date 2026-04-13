import { useState, useEffect, useRef } from 'react';
import './AdminPage.css';

export default function AdminPage({ onBack, onDocumentsUpdated }) {
  const [documents, setDocuments] = useState([]);
  const [newDoc, setNewDoc] = useState({ source: '', content: '' });
  const [department, setDepartment] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [uploadMode, setUploadMode] = useState('manual'); // 'manual' or 'file'
  const [pendingFiles, setPendingFiles] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const fetchDocuments = async () => {
    try {
      const response = await fetch("http://127.0.0.1:8000/admin/documents");
      if (response.ok) {
        const data = await response.json();
        setDocuments(data.documents);
      }
    } catch (error) {
      console.error('Failed to fetch documents:', error);
    }
  };

  // Fetch documents on mount
  useEffect(() => {
    fetchDocuments();
  }, []);

  const handleAddDocument = async () => {
    if (!newDoc.source.trim() || !newDoc.content.trim()) {
      setMessage('Please fill in both source and content');
      return;
    }

    setIsLoading(true);
    setMessage('');
    try {
      const response = await fetch("http://127.0.0.1:8000/admin/add-document", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          source: newDoc.source.trim(),
          content: newDoc.content.trim(),
          department: department.trim()
        })
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || "Failed to add document");
      }

      setMessage('✓ Document added successfully');
      setNewDoc({ source: '', content: '' });
      setDepartment('');
      
      // Refresh documents list
      setTimeout(() => fetchDocuments(), 500);
    } catch (error) {
      setMessage(`Error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteDocument = async (source) => {
    if (!window.confirm(`Delete document ${source}?`)) return;

    setIsLoading(true);
    setMessage('');
    try {
      const response = await fetch(`http://127.0.0.1:8000/admin/delete-document/${source}`, {
        method: "DELETE",
        headers: {
          "Content-Type": "application/json"
        }
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || "Failed to delete document");
      }

      setMessage(`✓ Document ${source} deleted`);
      setTimeout(() => fetchDocuments(), 500);
    } catch (error) {
      setMessage(`Error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFilesSelected = (event) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;
    setPendingFiles(prev => [...prev, ...Array.from(files)]);
    // Reset file input so same file can be re-selected
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleRemoveFile = (index) => {
    setPendingFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleConfirmUpload = async () => {
    if (pendingFiles.length === 0) return;

    setIsLoading(true);
    setMessage('');
    let successCount = 0;

    for (const file of pendingFiles) {
      try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('department', department.trim());

        const response = await fetch("http://127.0.0.1:8000/admin/upload-file", {
          method: "POST",
          body: formData
        });

        if (!response.ok) {
          const data = await response.json();
          throw new Error(data.detail || `Failed to upload ${file.name}`);
        }

        successCount++;
        setMessage(prev => prev ? `${prev}\n✓ ${file.name} uploaded` : `✓ ${file.name} uploaded`);
      } catch (error) {
        setMessage(prev => prev ? `${prev}\n❌ ${file.name}: ${error.message}` : `❌ ${file.name}: ${error.message}`);
      }
    }

    if (successCount > 0) {
      setTimeout(() => fetchDocuments(), 500);
    }
    setIsLoading(false);
    setPendingFiles([]);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      setPendingFiles(prev => [...prev, ...Array.from(files)]);
    }
  };

  return (
    <div className="admin-container">
      <header className="admin-header">
        <div className="admin-header-content">
          <h1>Admin: Manage Documents</h1>
          <button className="back-button" onClick={onBack}>← Back to Chat</button>
        </div>
      </header>

      <main className="admin-content">
        <section className="add-document-section">
          <h2>Add New Document</h2>
          
          <div className="mode-toggle">
            <button 
              className={`toggle-button ${uploadMode === 'manual' ? 'active' : ''}`}
              onClick={() => setUploadMode('manual')}
            >
              ✏️ Manual Entry
            </button>
            <button 
              className={`toggle-button ${uploadMode === 'file' ? 'active' : ''}`}
              onClick={() => setUploadMode('file')}
            >
              📁 Upload Files
            </button>
          </div>

          <div className="form-group">
            <label htmlFor="department">Department</label>
            <input
              id="department"
              type="text"
              placeholder="e.g., IT, HR, Finance"
              value={department}
              onChange={(e) => setDepartment(e.target.value)}
              disabled={isLoading}
            />
          </div>

          {uploadMode === 'manual' ? (
            <>
              <div className="form-group">
                <label htmlFor="source">Source ID</label>
                <input
                  id="source"
                  type="text"
                  placeholder="e.g., KB-006"
                  value={newDoc.source}
                  onChange={(e) => setNewDoc({ ...newDoc, source: e.target.value })}
                  disabled={isLoading}
                />
              </div>
              <div className="form-group">
                <label htmlFor="content">Content</label>
                <textarea
                  id="content"
                  placeholder="Enter the document content here..."
                  value={newDoc.content}
                  onChange={(e) => setNewDoc({ ...newDoc, content: e.target.value })}
                  rows="5"
                  disabled={isLoading}
                />
              </div>
              <button
                className="submit-button"
                onClick={handleAddDocument}
                disabled={isLoading}
              >
                {isLoading ? 'Adding...' : 'Add Document'}
              </button>
            </>
          ) : (
            <>
              <div
                className={`file-upload-area ${isDragging ? 'dragging' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                <label htmlFor="file-input" className="file-input-label">
                  <div className="upload-icon">{isDragging ? '📥' : '📄'}</div>
                  <p className="upload-text">{isDragging ? 'Drop files here' : 'Click to select files or drag & drop'}</p>
                  <p className="upload-hint">Supports: PDF, DOCX, MD, TXT, PNG, JPG, GIF</p>
                  <p className="upload-subliminal" style={{fontSize: '12px', color: 'var(--stone-gray)', marginTop: '8px'}}>
                    Images are processed with GPT-4o mini to extract conversation history
                  </p>
                </label>
                <input
                  ref={fileInputRef}
                  id="file-input"
                  type="file"
                  multiple
                  accept=".pdf,.docx,.doc,.md,.txt,.png,.jpg,.jpeg,.gif"
                  onChange={handleFilesSelected}
                  disabled={isLoading}
                  className="file-input"
                />
              </div>

              {pendingFiles.length > 0 && (
                <div className="pending-files">
                  <div className="pending-files-header">
                    <span>{pendingFiles.length} file{pendingFiles.length > 1 ? 's' : ''} selected</span>
                    <button className="clear-files-btn" onClick={() => setPendingFiles([])}>Clear all</button>
                  </div>
                  <div className="pending-files-list">
                    {pendingFiles.map((file, idx) => (
                      <div key={idx} className="pending-file-item">
                        <span className="pending-file-name">📄 {file.name}</span>
                        <span className="pending-file-size">{(file.size / 1024).toFixed(1)} KB</span>
                        <button className="remove-file-btn" onClick={() => handleRemoveFile(idx)}>✕</button>
                      </div>
                    ))}
                  </div>
                  <button
                    className="submit-button upload-confirm-btn"
                    onClick={handleConfirmUpload}
                    disabled={isLoading}
                  >
                    {isLoading ? 'Uploading...' : `Upload ${pendingFiles.length} File${pendingFiles.length > 1 ? 's' : ''}`}
                  </button>
                </div>
              )}
            </>
          )}

          {message && (
            <div className={`message ${message.includes('❌') || message.includes('Error') ? 'error' : 'success'}`}>
              {message.split('\n').map((line, i) => (
                <div key={i}>{line}</div>
              ))}
            </div>
          )}
        </section>

        <section className="documents-section">
          <h2>Existing Documents ({documents.length})</h2>
          {documents.length === 0 ? (
            <p className="empty-state">No documents yet. Add one to get started!</p>
          ) : (
            <div className="documents-list">
              {documents.map((doc, docIdx) => (
                <div key={`${doc.source}-${docIdx}`} className="document-card">
                  <div className="document-header">
                    <div
                      className="document-title-row"
                      onClick={() => {
                        setDocuments(prev =>
                          prev.map((d, i) =>
                            i === docIdx ? { ...d, expanded: !d.expanded } : d
                          )
                        );
                      }}
                    >
                      <span className="document-toggle">{doc.expanded ? '▼' : '▶'}</span>
                      <h3 className="document-source">{doc.source}</h3>
                      {doc.total_chunks > 1 && (
                        <span className="chunk-badge">Chunk {doc.chunk_index + 1}/{doc.total_chunks}</span>
                      )}
                      {doc.department && (
                        <span className="department-badge">{doc.department}</span>
                      )}
                    </div>
                    <button
                      className="delete-button"
                      onClick={() => handleDeleteDocument(doc.source)}
                      disabled={isLoading}
                    >
                      Delete
                    </button>
                  </div>
                  {doc.expanded && (
                    <p className="document-content">{doc.content}</p>
                  )}
                </div>
              ))}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
