// Used only for the single-container hosted build (FastAPI serves the UI and
// the API on one origin). apiUrl is relative so calls hit the same host.
export const environment = {
  production: true,
  apiUrl: '',
};
