// Make vtkPoints::New() hand out float64 points, in every library in the process.
//
// vtkPoints::New(dataType) asks the object factory first and only resets the
// type when the caller asked for something other than VTK_FLOAT, which is what
// the no-argument New() asks for. VMTK's filters and VTK's marching cubes build
// their output points with that default and so write float32 whatever the input
// was; with this override registered they write float64. A filter that sets its
// points' type explicitly afterwards (an OutputPointsPrecision of SINGLE) still
// gets what it asked for.
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <vtkObjectFactory.h>
#include <vtkPoints.h>
#include <vtkVersionMacros.h>

class vtkDoublePoints : public vtkPoints
{
public:
  vtkTypeMacro(vtkDoublePoints, vtkPoints);
  static vtkDoublePoints* New()
  {
    auto* result = new vtkDoublePoints;
    result->InitializeObjectBase();
    return result;
  }

protected:
  vtkDoublePoints() : vtkPoints(VTK_DOUBLE) {}
  ~vtkDoublePoints() override = default;
};

static vtkObject* CreateDoublePoints()
{
  return vtkDoublePoints::New();
}

class vtkDoublePointsFactory : public vtkObjectFactory
{
public:
  vtkTypeMacro(vtkDoublePointsFactory, vtkObjectFactory);
  static vtkDoublePointsFactory* New()
  {
    auto* result = new vtkDoublePointsFactory;
    result->InitializeObjectBase();
    return result;
  }
  const char* GetVTKSourceVersion() override { return VTK_SOURCE_VERSION; }
  const char* GetDescription() override { return "float64 default vtkPoints"; }

protected:
  vtkDoublePointsFactory()
  {
    this->RegisterOverride(
      "vtkPoints", "vtkDoublePoints", "float64 default vtkPoints", 1, CreateDoublePoints);
  }
};

static bool g_installed = false;

static PyObject* install(PyObject*, PyObject*)
{
  if (!g_installed)
  {
    vtkDoublePointsFactory* factory = vtkDoublePointsFactory::New();
    vtkObjectFactory::RegisterFactory(factory);
    factory->Delete();
    g_installed = true;
  }
  Py_RETURN_NONE;
}

static PyObject* installed(PyObject*, PyObject*)
{
  return PyBool_FromLong(g_installed ? 1 : 0);
}

static PyMethodDef methods[] = {
  { "install", install, METH_NOARGS, "Register the float64 vtkPoints factory (idempotent)." },
  { "installed", installed, METH_NOARGS, "Whether install() has run in this process." },
  { nullptr, nullptr, 0, nullptr },
};

static struct PyModuleDef module = { PyModuleDef_HEAD_INIT, "vtk_double_points",
  "Default vtkPoints to float64 process-wide.", -1, methods };

PyMODINIT_FUNC PyInit_vtk_double_points(void)
{
  return PyModule_Create(&module);
}
